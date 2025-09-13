#!/usr/bin/env python3
"""
Demo script for error detection API
Usage: python scripts/demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from scripts.create_dataset import create_real_dataset
from src.models.error_detector import ErrorDetector
from src.utils.logging import setup_logging

logger = structlog.get_logger()


async def run_demo():
    """Demo using the actual provided sample images"""
    print("="*70)
    print("ERROR DETECTION API - DEMO")
    print("="*70)

    setup_logging()

    # Create dataset from real images
    dataset = create_real_dataset()
    samples = dataset.samples

    print(f"\n🎯 Testing on {len(samples)} REAL student solutions...")
    print(f"   Questions: Q1-Q4 (Probability, Trigonometry, Algebra, Complex Numbers)")
    print(f"   Student Attempts: Attempt1-4 (Real handwritten solutions)")

    # Initialize error detector with hybrid approach
    detector = ErrorDetector(approach="hybrid")

    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*50}")
        print(f"🔍 ANALYZING SAMPLE {i}: {sample.id}")
        print(f"📚 Topic: {sample.metadata.get('topic', 'Unknown').title()}")
        print(f"📊 Difficulty: {sample.metadata.get('difficulty', 'Unknown')}")
        print(f"💡 Problem: {sample.metadata.get('question_text', 'See question image')}")
        print(f"{'='*50}")

        # Show ground truth for comparison
        if sample.ground_truth_error:
            print(f"🎯 GROUND TRUTH: ❌ Error Expected")
            print(f"   └─ {sample.ground_truth_error}")
            print(f"   └─ Correction: {sample.ground_truth_correction}")
        else:
            print(f"🎯 GROUND TRUTH: ✅ No errors expected")

        print(f"\n🤖 API ANALYSIS:")

        try:
            # For demo, we'll use the actual local file paths
            # In production, these would be uploaded to a web server
            question_path = sample.question_url.replace("file:///", "")
            solution_path = sample.solution_url.replace("file:///", "")
            question_url = f"file://{Path.cwd()}/{question_path}"
            solution_url = f"file://{Path.cwd()}/{solution_path}"

            # Run error detection
            result = await detector.detect_errors(
                question_url=question_url,
                solution_url=solution_url,
                context={
                    'sample_id': sample.id,
                    'topic': sample.metadata.get('topic'),
                    'ground_truth_has_error': bool(sample.ground_truth_error)
                }
            )

            # Display results
            if result.get('error'):
                print(f"   🔴 ERROR DETECTED: {result['error']}")
                print(f"   💡 Correction: {result.get('correction', 'N/A')}")
                print(f"   🔧 Hint: {result.get('hint', 'N/A')}")
                if result.get('y'):
                    print(f"   📍 Location (Y): {result.get('y')}")
            else:
                print(f"   ✅ NO ERRORS DETECTED")

            print(f"   ⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"   🏁 Solution complete: {result.get('solution_complete', False)}")
            print(f"   🎨 Contains diagrams: {result.get('contains_diagram', False)}")

            # Compare with ground truth
            gt_has_error = bool(sample.ground_truth_error)
            api_detected_error = bool(result.get('error'))

            print(f"\n📊 ACCURACY CHECK:")
            if gt_has_error == api_detected_error:
                print(f"   ✅ CORRECT: API {'detected' if api_detected_error else 'found no'} error (matches ground truth)")
            else:
                if gt_has_error and not api_detected_error:
                    print(f"   ❌ FALSE NEGATIVE: API missed an error")
                else:
                    print(f"   ❌ FALSE POSITIVE: API detected error where none exists")

            # Step-level analysis
            if result.get('solution_lines'):
                print(f"\n📝 EXTRACTED SOLUTION STEPS:")
                for j, line in enumerate(result['solution_lines'][:3], 1):  # Show first 3 steps
                    print(f"   {j}. {line}")
                if len(result['solution_lines']) > 3:
                    print(f"   ... and {len(result['solution_lines']) - 3} more steps")

        except Exception as e:
            print(f"   ❌ PROCESSING ERROR: {str(e)}")
            print(f"   (This might be due to mock/development setup)")

    # Summary
    print(f"\n{'='*70}")
    print("📋 DEMO SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Demonstrated real-world error detection on:")
    print(f"   • Probability problems (Bayes' theorem)")
    print(f"   • Trigonometry problems (angle of elevation)")
    print(f"   • Algebra problems (quadratic equations)")
    print(f"   • Complex number arithmetic")
    print(f"\n🎯 Key Features Showcased:")
    print(f"   • Multi-modal analysis (OCR + LLM + VLM)")
    print(f"   • Step-level error detection")
    print(f"   • Educational feedback (corrections + hints)")
    print(f"   • Handwriting processing")
    print(f"   • Real student work analysis")

    print(f"\n🚀 Next Steps:")
    print(f"   1. Set up API keys in .env file")
    print(f"   2. Run: make dev (start API server)")
    print(f"   3. Test with: curl -X POST http://localhost:8000/api/v1/detect-error")
    print(f"   4. Run full evaluation: python scripts/run_eval.py")


if __name__ == "__main__":
    asyncio.run(run_demo())