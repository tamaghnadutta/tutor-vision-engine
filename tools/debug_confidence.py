#!/usr/bin/env python3
"""
Debug script to trace confidence values through the system
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.error_detector import ErrorDetector

async def debug_confidence():
    """Debug confidence value flow"""

    print("🔍 DEBUGGING CONFIDENCE VALUES")
    print("=" * 50)

    # Test with a simple case
    detector = ErrorDetector(approach="openai")

    result = await detector.detect_errors(
        question_url="http://localhost:8080/data/sample_images/questions/Q1.jpeg",
        solution_url="http://localhost:8080/data/sample_images/attempts/Attempt1.jpeg",
        bounding_box={
            "minX": 0.05, "maxX": 0.65,
            "minY": 0.05, "maxY": 0.55
        },
        context={'debug': True}
    )

    print(f"📊 Final Result Confidence: {result.get('confidence', 'MISSING')}")
    print(f"📊 Processing Approach: {result.get('processing_approach', 'MISSING')}")
    print(f"📊 Has Error: {result.get('error') is not None}")

    # Let's examine the internal structure
    print(f"\n🔍 All result keys: {list(result.keys())}")

    # Print key confidence-related values
    for key in ['confidence', 'llm_used', 'processing_approach']:
        value = result.get(key)
        print(f"   {key}: {value} (type: {type(value)})")

if __name__ == "__main__":
    asyncio.run(debug_confidence())