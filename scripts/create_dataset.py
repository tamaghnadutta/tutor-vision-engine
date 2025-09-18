#!/usr/bin/env python3
"""
Create dataset from provided sample images
Usage: python scripts/create_dataset.py
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from src.data.dataset import ErrorDetectionDataset, DatasetSample
from src.utils.logging import setup_logging

logger = structlog.get_logger()


def create_real_dataset() -> ErrorDetectionDataset:
    """Create dataset from the provided real sample images"""

    samples = []

    # Sample 1: Probability Problem (Q1 + Attempt1)
    # Analyzing Attempt1: Student uses Bayes' theorem approach
    samples.append(DatasetSample(
        id="real_001",
        question_url="file:///data/sample_images/questions/Q1.jpeg",
        solution_url="file:///data/sample_images/attempts/Attempt1.jpeg",
        ground_truth_error=None,  # This appears to be a correct approach
        ground_truth_correction=None,
        ground_truth_hint=None,
        step_level_labels=[
            {"step": 1, "text": "k: student knows answer", "has_error": False},
            {"step": 2, "text": "g: student guessed the answer", "has_error": False},
            {"step": 3, "text": "c: student answer is correct", "has_error": False},
            {"step": 4, "text": "P(c|k) = 1", "has_error": False},
            {"step": 5, "text": "P(c|g) = 1/2", "has_error": False},
            {"step": 6, "text": "P(g|c) = 1/6", "has_error": False},
            {"step": 7, "text": "P(g|c) = P(c∩g)/P(c)", "has_error": False},
            {"step": 8, "text": "P(c∩g) = P(g)·P(c|g) = (1-P(k))·1/2", "has_error": False}
        ],
        is_noisy=False,
        metadata={
            "difficulty": "hard",
            "topic": "probability",
            "problem_type": "bayes_theorem",
            "question_text": "Probability problem about student guessing vs knowing answers",
            "has_correct_approach": True
        }
    ))

    # Sample 2: Trigonometry Problem (Q2 + Attempt2)
    # Analyzing Attempt2: Correct approach but arithmetic error
    samples.append(DatasetSample(
        id="real_002",
        question_url="file:///data/sample_images/questions/Q2.jpeg",
        solution_url="file:///data/sample_images/attempts/Attempt2.jpeg",
        ground_truth_error="Arithmetic error in final calculation",
        ground_truth_correction="47 × tan(35°) ≈ 47 × 0.7002 ≈ 32.9, not 33. The closest answer is b) 33",
        ground_truth_hint="Double-check your calculation of 47 × tan(35°). Use more precision in your trigonometric value.",
        step_level_labels=[
            {"step": 1, "text": "h = ?", "has_error": False},
            {"step": 2, "text": "tan(35°) = h/47", "has_error": False},
            {"step": 3, "text": "h = 47 × tan(35°)", "has_error": False},
            {"step": 4, "text": "h = 47 × 0.5 ≈ 33", "has_error": True, "error_type": "arithmetic"},
            {"step": 5, "text": "h = 33 feet", "has_error": True},
            {"step": 6, "text": "option b)", "has_error": False}
        ],
        is_noisy=False,
        metadata={
            "difficulty": "medium",
            "topic": "trigonometry",
            "problem_type": "angle_of_elevation",
            "question_text": "Find height of tree using angle of elevation",
            "correct_answer": "b) 33 (but calculation method has error)"
        }
    ))

    # Sample 3: Algebra Problem (Q3 + Attempt3)
    # Analyzing Attempt3: Error in solving (x-2)² = 36
    samples.append(DatasetSample(
        id="real_003",
        question_url="file:///data/sample_images/questions/Q3.jpeg",
        solution_url="file:///data/sample_images/attempts/Attempt3.jpeg",
        ground_truth_error="Incorrect solution to square root equation",
        ground_truth_correction="From (x-2)² = 36, we get x-2 = ±6, so x = 2±6, giving x = 8 or x = -4",
        ground_truth_hint="When solving (x-2)² = 36, remember to consider both positive and negative square roots",
        step_level_labels=[
            {"step": 1, "text": "x = ?", "has_error": False},
            {"step": 2, "text": "3(x-2)² = 75", "has_error": False},
            {"step": 3, "text": "(x-2)² = 25", "has_error": True, "error_type": "arithmetic"},  # Should be 25, not 36
            {"step": 4, "text": "(x-2)² = (6)² ⟹ x-2 = +6", "has_error": True, "error_type": "incomplete"},  # Missing ±
            {"step": 5, "text": "x = 2 + 5", "has_error": True},  # Should be 2 + 5 = 7 or 2 - 5 = -3
            {"step": 6, "text": "x = 7", "has_error": True}  # Missing second solution
        ],
        is_noisy=False,
        metadata={
            "difficulty": "medium",
            "topic": "algebra",
            "problem_type": "quadratic_equations",
            "question_text": "Solve 3(x-2)² = 75",
            "correct_answer": "x = 7 or x = -3 (student got partial answer)"
        }
    ))

    # Sample 4: Complex Numbers Problem (Q4 + Attempt4)
    # Analyzing Attempt4: Complex computation with potential errors
    samples.append(DatasetSample(
        id="real_004",
        question_url="file:///data/sample_images/questions/Q4.jpeg",
        solution_url="file:///data/sample_images/attempts/Attempt4.jpeg",
        ground_truth_error="Possible error in complex number calculation",
        ground_truth_correction="Need to verify the expansion and substitution steps carefully",
        ground_truth_hint="Check each step of the complex number arithmetic, especially the conjugate calculation",
        step_level_labels=[
            {"step": 1, "text": "1) expand expression", "has_error": False},
            {"step": 2, "text": "(z+3)(z̄+3) = z²z̄ + 3z + 9 = |z|² + 3(z+z̄) + 9", "has_error": False},
            {"step": 3, "text": "2) Calculate components with z = 1+2i", "has_error": False},
            {"step": 4, "text": "|z|² = 1² - 2² = 1 + (4) = -3", "has_error": True, "error_type": "calculation"},  # Should be 1² + 2² = 5
            {"step": 5, "text": "z + z̄ = (1+2i) + (1-2i) = 2", "has_error": False},
            {"step": 6, "text": "3) Substitute", "has_error": False},
            {"step": 7, "text": "= -3 + 3(2) + 9 = 12", "has_error": True}  # Wrong due to error in step 4
        ],
        is_noisy=False,
        metadata={
            "difficulty": "hard",
            "topic": "complex_numbers",
            "problem_type": "complex_arithmetic",
            "question_text": "Complex number conjugate and modulus calculation",
            "has_calculation_errors": True
        }
    ))

    # Create dataset object
    dataset = ErrorDetectionDataset.__new__(ErrorDetectionDataset)
    dataset.samples = samples
    return dataset


def convert_to_web_urls(dataset: ErrorDetectionDataset) -> ErrorDetectionDataset:
    """Convert file:// URLs to proper web URLs for the actual images"""

    # Use localhost URLs to serve images via local HTTP server
    # Start HTTP server with: python -m http.server 8080
    base_url = "http://localhost:8080/data/sample_images"

    for sample in dataset.samples:
        # Convert question URLs
        if sample.question_url.startswith("file:///data/sample_images/questions/"):
            filename = sample.question_url.split("/")[-1]
            sample.question_url = f"{base_url}/questions/{filename}"

        # Convert solution URLs
        if sample.solution_url.startswith("file:///data/sample_images/attempts/"):
            filename = sample.solution_url.split("/")[-1]
            sample.solution_url = f"{base_url}/attempts/{filename}"

    return dataset


def main():
    """Create real dataset from provided images"""
    setup_logging()

    logger.info("Creating real dataset from provided sample images")

    # Create dataset
    dataset = create_real_dataset()

    # Convert to web URLs (in production, you'd actually upload the images)
    dataset = convert_to_web_urls(dataset)

    # Save dataset
    output_path = "./data/real_eval_dataset.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_dataset(output_path)

    # Print analysis summary
    total_samples = len(dataset.samples)
    error_samples = len([s for s in dataset.samples if s.ground_truth_error])
    step_lines = sum(len(s.step_level_labels) for s in dataset.samples)

    print(f"\n✅ Real Dataset Created!")
    print(f"📁 Output: {output_path}")
    print(f"📊 Total samples: {total_samples}")
    print(f"❌ Samples with errors: {error_samples}")
    print(f"📝 Total step lines: {step_lines}")

    # Topic breakdown
    topics = {}
    for sample in dataset.samples:
        topic = sample.metadata.get('topic', 'unknown')
        topics[topic] = topics.get(topic, 0) + 1

    print(f"\n📚 Topic breakdown:")
    for topic, count in topics.items():
        print(f"  {topic}: {count}")

    # Error analysis
    print(f"\n🔍 Error Analysis:")
    for sample in dataset.samples:
        if sample.ground_truth_error:
            print(f"  {sample.id} ({sample.metadata.get('topic')}): {sample.ground_truth_error}")
        else:
            print(f"  {sample.id} ({sample.metadata.get('topic')}): ✅ No errors detected")

    logger.info("Real dataset creation completed successfully")


if __name__ == "__main__":
    main()