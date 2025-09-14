#!/usr/bin/env python3
"""
Use GPT-4V to analyze solution images and determine optimal bounding boxes
"""

import asyncio
import aiohttp
import base64
import json
import os
from pathlib import Path
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found in environment variables")
    print("Make sure your .env file contains OPENAI_API_KEY")
    exit(1)

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def analyze_image_for_bounding_boxes(session: aiohttp.ClientSession, image_path: str, image_name: str) -> dict:
    """
    Use GPT-4V to analyze an image and suggest optimal bounding boxes
    """
    print(f"\n🔍 Analyzing {image_name} with GPT-4V...")

    # Encode image
    base64_image = encode_image_to_base64(image_path)

    # Create the prompt for bounding box analysis
    prompt = """
    Analyze this handwritten mathematical solution image and provide optimal bounding boxes for error detection.

    Please identify:
    1. The main work area (where most calculations are done)
    2. Individual calculation steps or lines
    3. Any areas with potential errors or corrections
    4. Key formula or equation areas

    For each identified area, provide bounding box coordinates as normalized values (0.0 to 1.0) where:
    - minX, maxX: horizontal coordinates (0.0 = left edge, 1.0 = right edge)
    - minY, maxY: vertical coordinates (0.0 = top edge, 1.0 = bottom edge)

    Return your analysis in this JSON format:
    {
        "image_dimensions": "describe the image layout",
        "content_type": "describe what type of math problem this is",
        "bounding_boxes": [
            {
                "name": "descriptive name",
                "description": "what this area contains",
                "coordinates": {"minX": 0.0, "maxX": 1.0, "minY": 0.0, "maxY": 1.0},
                "priority": "high/medium/low",
                "recommended_for_error_detection": true/false
            }
        ],
        "analysis": "overall analysis and recommendations"
    }

    Focus on areas most likely to contain mathematical errors or key calculation steps.
    """

    payload = {
        "model": "gpt-4o",
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
        "max_tokens": 1500,
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

                        print(f"✅ Analysis completed for {image_name}")
                        print(f"   Content type: {analysis_data.get('content_type', 'Unknown')}")
                        print(f"   Found {len(analysis_data.get('bounding_boxes', []))} bounding box recommendations")

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

async def analyze_all_images():
    """Analyze all sample images for bounding boxes"""

    print("🚀 ANALYZING SAMPLE IMAGES FOR OPTIMAL BOUNDING BOXES")
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
                result = await analyze_image_for_bounding_boxes(session, image_path, image_name)
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
    output_file = "gpt4v_bounding_box_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Analysis results saved to {output_file}")

    # Summary
    successful = [r for r in results if r.get("success", False)]
    print(f"\n📊 SUMMARY:")
    print(f"✅ Successfully analyzed: {len(successful)}/{len(results)} images")

    if successful:
        print(f"\n🎯 RECOMMENDED BOUNDING BOXES:")
        for result in successful:
            analysis = result.get("analysis", {})
            bounding_boxes = analysis.get("bounding_boxes", [])

            print(f"\n📋 {result['image_name']}:")
            print(f"   Content: {analysis.get('content_type', 'Unknown')}")

            high_priority = [bb for bb in bounding_boxes if bb.get("priority") == "high"]
            if high_priority:
                print(f"   🔥 High priority areas:")
                for bb in high_priority:
                    coords = bb.get("coordinates", {})
                    print(f"      • {bb.get('name', 'Unnamed')}: {coords}")
                    print(f"        {bb.get('description', 'No description')}")

    return results

async def create_test_cases_from_analysis():
    """Create test cases based on GPT-4V analysis"""

    # Load the analysis results
    try:
        with open("gpt4v_bounding_box_analysis.json", "r") as f:
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
        bounding_boxes = analysis.get("bounding_boxes", [])

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

        # Create test cases for high and medium priority bounding boxes
        for bb in bounding_boxes:
            if bb.get("recommended_for_error_detection", False):
                test_case = {
                    "name": f"{image_name} - {bb.get('name', 'Area')}",
                    "question_url": question_url_map.get(image_name),
                    "solution_url": solution_url_map.get(image_name),
                    "bounding_box": bb.get("coordinates", {}),
                    "description": f"{bb.get('description', '')} (Priority: {bb.get('priority', 'unknown')})",
                    "gpt4v_analysis": {
                        "content_type": analysis.get("content_type"),
                        "priority": bb.get("priority"),
                        "recommended": bb.get("recommended_for_error_detection")
                    }
                }
                test_cases.append(test_case)

    # Save test cases
    with open("gpt4v_test_cases.json", "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"\n🧪 Created {len(test_cases)} test cases from GPT-4V analysis")
    print("💾 Test cases saved to gpt4v_test_cases.json")

    return test_cases

async def main():
    """Main function"""
    print("Choose an option:")
    print("1. Analyze images with GPT-4V")
    print("2. Create test cases from existing analysis")
    print("3. Both")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice in ["1", "3"]:
        await analyze_all_images()

    if choice in ["2", "3"]:
        await create_test_cases_from_analysis()

if __name__ == "__main__":
    asyncio.run(main())