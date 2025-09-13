#!/usr/bin/env python3
"""
Test the /detect-error endpoint with complete student solution area bounding boxes
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "test-api-key-123")

def load_complete_solution_test_cases() -> List[Dict[str, Any]]:
    """Load test cases for complete solution areas"""
    try:
        with open("complete_solution_test_cases.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Complete solution test cases not found. Run analyze_full_solution_areas.py first.")
        return []

async def test_detect_error_endpoint(session: aiohttp.ClientSession, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test the detect-error endpoint with a complete solution area bounding box
    """
    print(f"\n🧪 Testing: {test_case['name']}")
    print(f"   Description: {test_case['description']}")

    bbox = test_case['bounding_box']
    print(f"   Complete Solution Area: ({bbox['minX']:.3f}, {bbox['minY']:.3f}) → ({bbox['maxX']:.3f}, {bbox['maxY']:.3f})")

    gpt4v_info = test_case.get('gpt4v_analysis', {})
    print(f"   Content Type: {gpt4v_info.get('content_type', 'Unknown')}")
    print(f"   Coverage: {gpt4v_info.get('coverage_description', 'No description')[:80]}...")
    print(f"   Confidence: {gpt4v_info.get('confidence', 'unknown')}")

    # Calculate area coverage for reference
    width = bbox['maxX'] - bbox['minX']
    height = bbox['maxY'] - bbox['minY']
    area_percentage = width * height * 100
    print(f"   Area Coverage: {area_percentage:.1f}% of image")

    # Prepare the request payload
    payload = {
        "question_url": test_case["question_url"],
        "solution_url": test_case["solution_url"],
        "bounding_box": test_case["bounding_box"],
        "user_id": "complete_solution_test_user",
        "session_id": f"complete_session_{int(time.time())}",
        "question_id": test_case["name"].replace(" ", "_").lower()
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    start_time = time.time()

    try:
        async with session.post(
            f"{API_BASE_URL}/api/v1/detect-error",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120)  # 2 minute timeout
        ) as response:

            duration = time.time() - start_time

            if response.status == 200:
                result = await response.json()

                print(f"   ✅ Success: {duration:.2f}s")
                print(f"   Has error: {result.get('error') is not None}")
                print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
                print(f"   Processing approach: {result.get('processing_approach', 'unknown')}")
                print(f"   Solution complete: {result.get('solution_complete', False)}")

                if result.get('error'):
                    error_text = result['error']
                    print(f"   🚨 Error found: {error_text[:120]}{'...' if len(error_text) > 120 else ''}")

                if result.get('y'):
                    print(f"   📍 Error Y position: {result['y']}")

                if result.get('correction'):
                    correction_text = result['correction']
                    print(f"   🔧 Correction: {correction_text[:120]}{'...' if len(correction_text) > 120 else ''}")

                return {
                    "test_case": test_case["name"],
                    "success": True,
                    "duration": duration,
                    "result": result,
                    "gpt4v_analysis": gpt4v_info,
                    "area_coverage": area_percentage
                }

            else:
                error_text = await response.text()
                print(f"   ❌ HTTP Error {response.status}: {error_text}")

                return {
                    "test_case": test_case["name"],
                    "success": False,
                    "duration": duration,
                    "error": f"HTTP {response.status}: {error_text}",
                    "gpt4v_analysis": gpt4v_info,
                    "area_coverage": area_percentage
                }

    except asyncio.TimeoutError:
        duration = time.time() - start_time
        print(f"   ⏰ Timeout: {duration:.2f}s")
        return {
            "test_case": test_case["name"],
            "success": False,
            "duration": duration,
            "error": "Request timeout",
            "gpt4v_analysis": gpt4v_info,
            "area_coverage": area_percentage
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"   ❌ Error: {duration:.2f}s - {str(e)}")
        return {
            "test_case": test_case["name"],
            "success": False,
            "duration": duration,
            "error": str(e),
            "gpt4v_analysis": gpt4v_info,
            "area_coverage": area_percentage
        }

async def test_all_complete_solution_areas():
    """
    Test all complete solution area bounding boxes
    """
    print("🚀 TESTING COMPLETE STUDENT SOLUTION AREAS")
    print("=" * 80)

    test_cases = load_complete_solution_test_cases()
    if not test_cases:
        return []

    print(f"📋 Testing {len(test_cases)} complete solution areas:")
    for i, case in enumerate(test_cases, 1):
        gpt4v_info = case.get("gpt4v_analysis", {})
        bbox = case["bounding_box"]
        width = bbox['maxX'] - bbox['minX']
        height = bbox['maxY'] - bbox['minY']
        area = width * height * 100
        print(f"   {i}. {case['name']} ({area:.1f}% coverage)")

    async with aiohttp.ClientSession() as session:
        # Test API health first
        try:
            async with session.get(f"{API_BASE_URL}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"\n✅ API Health Check: {health_data}")
                else:
                    print(f"\n⚠️ API Health Check failed: {response.status}")
        except Exception as e:
            print(f"\n❌ Cannot connect to API: {e}")
            return []

        print()

        # Run all test cases
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"TEST {i}/{len(test_cases)}")
            print(f"{'='*80}")

            result = await test_detect_error_endpoint(session, test_case)
            results.append(result)

            # Wait between tests to avoid rate limiting
            if i < len(test_cases):
                print(f"\n   💤 Waiting 3s before next test...")
                await asyncio.sleep(3)

        return results

async def analyze_complete_solution_results(results: List[Dict[str, Any]]):
    """Analyze and summarize complete solution area test results"""

    print(f"\n{'='*80}")
    print("📊 COMPLETE SOLUTION AREA RESULTS ANALYSIS")
    print(f"{'='*80}")

    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]

    print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed tests: {len(failed_tests)}")

    if successful_tests:
        avg_duration = sum(r["duration"] for r in successful_tests) / len(successful_tests)
        max_duration = max(r["duration"] for r in successful_tests)
        min_duration = min(r["duration"] for r in successful_tests)

        print(f"⏱️ Latency Stats: avg={avg_duration:.2f}s, min={min_duration:.2f}s, max={max_duration:.2f}s")

        # Analyze by content type
        by_content_type = {}
        for result in successful_tests:
            content_type = result.get("gpt4v_analysis", {}).get("content_type", "Unknown")
            if content_type not in by_content_type:
                by_content_type[content_type] = []
            by_content_type[content_type].append(result)

        print(f"\n📈 Results by Content Type:")
        for content_type, type_results in by_content_type.items():
            avg_conf = sum(r["result"].get("confidence", 0) for r in type_results) / len(type_results)
            error_count = sum(1 for r in type_results if r["result"].get("error"))
            avg_time = sum(r["duration"] for r in type_results) / len(type_results)
            avg_coverage = sum(r.get("area_coverage", 0) for r in type_results) / len(type_results)

            print(f"   📚 {content_type}:")
            print(f"      • Tests: {len(type_results)}")
            print(f"      • Avg confidence: {avg_conf:.3f}")
            print(f"      • Errors detected: {error_count}/{len(type_results)}")
            print(f"      • Avg time: {avg_time:.2f}s")
            print(f"      • Avg area coverage: {avg_coverage:.1f}%")

        # Compare coverage efficiency
        coverage_efficiency = {}
        for result in successful_tests:
            coverage = result.get("area_coverage", 0)
            duration = result["duration"]
            confidence = result["result"].get("confidence", 0)

            efficiency_score = (confidence * 100) / (duration * coverage) if coverage > 0 and duration > 0 else 0
            coverage_efficiency[result["test_case"]] = {
                "coverage": coverage,
                "duration": duration,
                "confidence": confidence,
                "efficiency": efficiency_score
            }

        print(f"\n📐 Coverage Efficiency Analysis:")
        sorted_efficiency = sorted(coverage_efficiency.items(), key=lambda x: x[1]["efficiency"], reverse=True)
        for test_name, metrics in sorted_efficiency:
            print(f"   📊 {test_name}:")
            print(f"      • Coverage: {metrics['coverage']:.1f}%, Duration: {metrics['duration']:.2f}s, Confidence: {metrics['confidence']:.3f}")
            print(f"      • Efficiency score: {metrics['efficiency']:.4f}")

        print(f"\n🎯 Individual Test Details:")
        for result in successful_tests:
            test_result = result["result"]
            duration = result["duration"]
            confidence = test_result.get("confidence", 0.0)
            has_error = test_result.get("error") is not None
            coverage = result.get("area_coverage", 0)

            status_icon = "🚨" if has_error else "✅"
            print(f"   {status_icon} {result['test_case']}: {duration:.2f}s, conf={confidence:.3f}, coverage={coverage:.1f}%")

    if failed_tests:
        print(f"\n❌ Failed Test Details:")
        for result in failed_tests:
            print(f"   • {result['test_case']}: {result['error']}")

    return {
        "total_tests": len(results),
        "successful": len(successful_tests),
        "failed": len(failed_tests),
        "avg_duration": avg_duration if successful_tests else 0,
        "results": results
    }

async def main():
    """Main test function"""
    results = await test_all_complete_solution_areas()

    if results:
        analysis = await analyze_complete_solution_results(results)

        # Save detailed results
        timestamp = int(time.time())
        results_file = f"complete_solution_test_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "test_type": "complete_solution_areas",
                "summary": analysis,
                "detailed_results": results
            }, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to {results_file}")

        print(f"\n🎉 Complete solution area testing completed!")
        print(f"   Success rate: {analysis['successful']}/{analysis['total_tests']} ({analysis['successful']/analysis['total_tests']*100:.1f}%)")
        if analysis['avg_duration'] > 0:
            print(f"   Average latency: {analysis['avg_duration']:.2f}s")

        print(f"\n📐 Key Insights:")
        if results:
            avg_coverage = sum(r.get("area_coverage", 0) for r in results) / len(results)
            print(f"   • Average area coverage: {avg_coverage:.1f}% of image")
            print(f"   • These bounding boxes represent COMPLETE student work areas")
            print(f"   • Perfect for analyzing entire solution approaches")

if __name__ == "__main__":
    asyncio.run(main())