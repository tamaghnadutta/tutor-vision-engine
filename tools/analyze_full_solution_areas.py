#!/usr/bin/env python3
"""
Use GPT-4V to identify the complete student solution area bounding boxes
(not individual lines, but the entire handwritten work area)
"""

import asyncio
import aiohttp
import base64
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found in environment variables")
    exit(1)

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def analyze_full_solution_area(session: aiohttp.ClientSession, image_path: str, image_name: str) -> dict:
    """
    Use GPT-4V to identify the complete student solution area bounding box
    """
    print(f"\n🔍 Analyzing complete solution area for {image_name}...")

    # Encode image
    base64_image = encode_image_to_base64(image_path)

    # Create the prompt for full solution area analysis
    prompt = """
    Analyze this handwritten mathematical solution image and identify the COMPLETE STUDENT SOLUTION AREA.

    I need the bounding box that encompasses the ENTIRE handwritten work area - not individual lines or steps, but the full rectangular area that contains ALL the student's handwritten work.

    Please identify:
    1. The outermost boundaries of all handwritten content
    2. Include any mathematical notation, equations, calculations, and text
    3. Exclude any blank space or margins that don't contain student work
    4. Make sure to capture the full height and width of the complete solution

    Provide bounding box coordinates as normalized values (0.0 to 1.0) where:
    - minX, maxX: horizontal coordinates (0.0 = left edge, 1.0 = right edge)
    - minY, maxY: vertical coordinates (0.0 = top edge, 1.0 = bottom edge)

    Return your analysis in this JSON format:
    {
        "image_analysis": "describe what you see in the image",
        "content_type": "what type of math problem this is",
        "solution_area_description": "describe the boundaries of the complete handwritten area",
        "complete_solution_bounding_box": {
            "minX": 0.0,
            "maxX": 1.0,
            "minY": 0.0,
            "maxY": 1.0
        },
        "confidence": "high/medium/low - how confident you are about these boundaries",
        "notes": "any important observations about the solution area"
    }

    Focus on capturing the ENTIRE student work area, not individual components.
    """

    payload = {
        "model": "gpt-4o",  # Using latest GPT-4o for vision analysis
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.1
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:

            if response.status == 200:
                result = await response.json()
                analysis_text = result["choices"][0]["message"]["content"]

                # Try to extract JSON from the response
                try:
                    # Find JSON in the response
                    json_start = analysis_text.find('{')
                    json_end = analysis_text.rfind('}') + 1

                    if json_start != -1 and json_end > json_start:
                        json_str = analysis_text[json_start:json_end]
                        analysis_data = json.loads(json_str)

                        bbox = analysis_data.get('complete_solution_bounding_box', {})
                        print(f"✅ Full solution area analyzed for {image_name}")
                        print(f"   Content type: {analysis_data.get('content_type', 'Unknown')}")
                        print(f"   Bounding box: ({bbox.get('minX', 0):.3f}, {bbox.get('minY', 0):.3f}) → ({bbox.get('maxX', 1):.3f}, {bbox.get('maxY', 1):.3f})")
                        print(f"   Confidence: {analysis_data.get('confidence', 'unknown')}")

                        return {
                            "image_name": image_name,
                            "image_path": image_path,
                            "success": True,
                            "analysis": analysis_data
                        }
                    else:
                        print(f"❌ Could not extract JSON from GPT-4V response for {image_name}")
                        return {
                            "image_name": image_name,
                            "success": False,
                            "error": "Could not parse JSON response",
                            "raw_response": analysis_text
                        }

                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing error for {image_name}: {e}")
                    return {
                        "image_name": image_name,
                        "success": False,
                        "error": f"JSON parsing error: {e}",
                        "raw_response": analysis_text
                    }
            else:
                error_text = await response.text()
                print(f"❌ OpenAI API error for {image_name}: {response.status} - {error_text}")
                return {
                    "image_name": image_name,
                    "success": False,
                    "error": f"API error {response.status}: {error_text}"
                }

    except Exception as e:
        print(f"❌ Exception analyzing {image_name}: {e}")
        return {
            "image_name": image_name,
            "success": False,
            "error": str(e)
        }

async def analyze_all_solution_areas():
    """Analyze all sample images for complete solution area bounding boxes"""

    print("🚀 ANALYZING COMPLETE STUDENT SOLUTION AREAS")
    print("=" * 60)

    # Sample images to analyze
    image_files = [
        ("data/sample_images/attempts/Attempt1.jpeg", "Attempt1"),
        ("data/sample_images/attempts/Attempt2.jpeg", "Attempt2"),
        ("data/sample_images/attempts/Attempt3.jpeg", "Attempt3"),
        ("data/sample_images/attempts/Attempt4.jpeg", "Attempt4")
    ]

    results = []

    async with aiohttp.ClientSession() as session:
        for image_path, image_name in image_files:
            if Path(image_path).exists():
                result = await analyze_full_solution_area(session, image_path, image_name)
                results.append(result)

                # Wait between requests to respect rate limits
                await asyncio.sleep(2)
            else:
                print(f"⚠️ Image not found: {image_path}")
                results.append({
                    "image_name": image_name,
                    "success": False,
                    "error": "Image file not found"
                })

    # Save results
    output_file = "complete_solution_areas.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Complete solution area analysis saved to {output_file}")

    # Summary
    successful = [r for r in results if r.get("success", False)]
    print(f"\n📊 SUMMARY:")
    print(f"✅ Successfully analyzed: {len(successful)}/{len(results)} images")

    if successful:
        print(f"\n📐 COMPLETE SOLUTION AREA BOUNDING BOXES:")
        for result in successful:
            analysis = result.get("analysis", {})
            bbox = analysis.get("complete_solution_bounding_box", {})

            print(f"\n📋 {result['image_name']}:")
            print(f"   Content: {analysis.get('content_type', 'Unknown')}")
            print(f"   Bounding Box: ({bbox.get('minX', 0):.3f}, {bbox.get('minY', 0):.3f}) → ({bbox.get('maxX', 1):.3f}, {bbox.get('maxY', 1):.3f})")
            print(f"   Coverage: {analysis.get('solution_area_description', 'No description')}")
            print(f"   Confidence: {analysis.get('confidence', 'unknown')}")

    return results

async def create_full_solution_test_cases():
    """Create test cases with complete solution area bounding boxes"""

    # Load the analysis results
    try:
        with open("complete_solution_areas.json", "r") as f:
            analysis_results = json.load(f)
    except FileNotFoundError:
        print("❌ No analysis results found. Run analysis first.")
        return

    test_cases = []

    for result in analysis_results:
        if not result.get("success", False):
            continue

        image_name = result["image_name"]
        analysis = result.get("analysis", {})
        bbox = analysis.get("complete_solution_bounding_box", {})

        # Map image names to URLs
        question_url_map = {
            "Attempt1": "http://localhost:8080/data/sample_images/questions/Q1.jpeg",
            "Attempt2": "http://localhost:8080/data/sample_images/questions/Q2.jpeg",
            "Attempt3": "http://localhost:8080/data/sample_images/questions/Q3.jpeg",
            "Attempt4": "http://localhost:8080/data/sample_images/questions/Q4.jpeg"
        }

        solution_url_map = {
            "Attempt1": "http://localhost:8080/data/sample_images/attempts/Attempt1.jpeg",
            "Attempt2": "http://localhost:8080/data/sample_images/attempts/Attempt2.jpeg",
            "Attempt3": "http://localhost:8080/data/sample_images/attempts/Attempt3.jpeg",
            "Attempt4": "http://localhost:8080/data/sample_images/attempts/Attempt4.jpeg"
        }

        # Create test case for complete solution area
        test_case = {
            "name": f"{image_name} - Complete Solution Area",
            "question_url": question_url_map.get(image_name),
            "solution_url": solution_url_map.get(image_name),
            "bounding_box": bbox,
            "description": f"Complete handwritten solution area for {analysis.get('content_type', 'math problem')}",
            "gpt4v_analysis": {
                "content_type": analysis.get("content_type"),
                "coverage_description": analysis.get("solution_area_description"),
                "confidence": analysis.get("confidence"),
                "analysis_type": "complete_solution_area"
            }
        }
        test_cases.append(test_case)

    # Save test cases
    with open("complete_solution_test_cases.json", "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"\n🧪 Created {len(test_cases)} test cases for complete solution areas")
    print("💾 Test cases saved to complete_solution_test_cases.json")

    return test_cases

async def main():
    """Main function"""
    print("Choose an option:")
    print("1. Analyze complete solution areas with GPT-4V")
    print("2. Create test cases from existing analysis")
    print("3. Both")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice in ["1", "3"]:
        await analyze_all_solution_areas()

    if choice in ["2", "3"]:
        await create_full_solution_test_cases()

if __name__ == "__main__":
    asyncio.run(main())