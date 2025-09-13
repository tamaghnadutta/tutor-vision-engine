#!/usr/bin/env python3
"""
Evaluation harness for error detection models
Usage: python scripts/run_eval.py
"""

import asyncio
import json
import time
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import tabulate

from scripts.create_dataset import create_real_dataset, convert_to_web_urls
from src.config.settings import get_settings
from src.data.dataset import ErrorDetectionDataset
from src.eval.evaluator import ErrorDetectionEvaluator, EvaluationMetrics
from src.models.error_detector import ErrorDetector
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class EvaluationHarness:
    """Main evaluation harness for real data"""

    def __init__(self, args):
        self.args = args
        self.settings = get_settings()
        self.results = {}

    async def run_evaluation(self) -> dict:
        """Run complete evaluation pipeline"""
        logger.info("Starting evaluation harness with real data")

        # 1. Load dataset
        dataset_path = self.args.dataset_path
        if not Path(dataset_path).exists():
            logger.error(f"Dataset not found at {dataset_path}")
            return {}

        # 2. Initialize evaluator
        evaluator = ErrorDetectionEvaluator(dataset_path)

        # 3. Run baseline model (OCR+LLM)
        logger.info("Evaluating baseline model (OCR+LLM)")
        baseline_detector = ErrorDetector(approach="auto")  # Use configured provider
        baseline_metrics = await evaluator.evaluate_model(
            baseline_detector,
            "baseline_ocr_llm",
            max_concurrent=self.args.max_concurrent
        )

        # 4. Run improved model (Hybrid)
        logger.info("Evaluating improved model (Hybrid)")
        improved_detector = ErrorDetector(approach="auto")  # Use configured provider
        improved_metrics = await evaluator.evaluate_model(
            improved_detector,
            "improved_hybrid",
            max_concurrent=self.args.max_concurrent
        )

        # 5. Calculate improvements
        improvements = evaluator.compare_models(baseline_metrics, improved_metrics)

        # 6. Compile results
        self.results = {
            'baseline_metrics': baseline_metrics,
            'improved_metrics': improved_metrics,
            'improvements': improvements,
            'evaluation_metadata': {
                'dataset_path': dataset_path,
                'total_samples': len(evaluator.dataset.get_samples()),
                'max_concurrent': self.args.max_concurrent,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        }

        # 7. Save results
        if self.args.output:
            self._save_results(self.args.output)

        # 8. Print metrics table
        self._print_metrics_table()

        # 9. Export per-case results
        if self.args.export_cases:
            evaluator.save_results(self.args.export_cases)

        logger.info("Evaluation completed successfully")
        return self.results

    def _print_metrics_table(self):
        """Print formatted metrics table"""
        print("\n" + "="*80)
        print("ERROR DETECTION API - EVALUATION RESULTS")
        print("="*80)

        # Main metrics comparison
        headers = ["Metric", "Baseline (OCR+LLM)", "Improved (Hybrid)", "Improvement"]
        rows = [
            ["Accuracy", f"{self.results['baseline_metrics'].accuracy:.3f}",
             f"{self.results['improved_metrics'].accuracy:.3f}",
             f"{self.results['improvements']['accuracy_improvement']:+.3f}"],
            ["F1 Score", f"{self.results['baseline_metrics'].f1_score:.3f}",
             f"{self.results['improved_metrics'].f1_score:.3f}",
             f"{self.results['improvements']['f1_improvement']:+.3f}"],
            ["Step-Level Accuracy", f"{self.results['baseline_metrics'].step_level_accuracy:.3f}",
             f"{self.results['improved_metrics'].step_level_accuracy:.3f}",
             f"{self.results['improvements']['step_level_accuracy_improvement']:+.3f}"],
            ["Error Detection Rate", f"{self.results['baseline_metrics'].error_detection_rate:.3f}",
             f"{self.results['improved_metrics'].error_detection_rate:.3f}", "N/A"],
        ]

        print("\nCore Performance Metrics:")
        print(tabulate.tabulate(rows, headers=headers, tablefmt="grid"))

        # Latency metrics
        print("\nLatency Metrics (seconds):")
        latency_headers = ["Percentile", "Baseline", "Improved", "Improvement"]
        latency_rows = [
            ["p50", f"{self.results['baseline_metrics'].latency_p50:.2f}",
             f"{self.results['improved_metrics'].latency_p50:.2f}",
             f"{self.results['baseline_metrics'].latency_p50 - self.results['improved_metrics'].latency_p50:+.2f}"],
            ["p90", f"{self.results['baseline_metrics'].latency_p90:.2f}",
             f"{self.results['improved_metrics'].latency_p90:.2f}",
             f"{self.results['baseline_metrics'].latency_p90 - self.results['improved_metrics'].latency_p90:+.2f}"],
            ["p95", f"{self.results['baseline_metrics'].latency_p95:.2f}",
             f"{self.results['improved_metrics'].latency_p95:.2f}",
             f"{self.results['improvements']['latency_p95_improvement']:+.2f}"],
        ]
        print(tabulate.tabulate(latency_rows, headers=latency_headers, tablefmt="grid"))

        # Cost estimation
        print("\nCost Estimation (per 100 requests):")
        baseline_cost = self._estimate_cost("ocr_llm", 100)
        improved_cost = self._estimate_cost("hybrid", 100)
        cost_headers = ["Model", "Estimated Cost", "Notes"]
        cost_rows = [
            ["Baseline (OCR+LLM)", f"${baseline_cost:.2f}", "Gemini 2.5 Flash (thinking_budget=0)"],
            ["Improved (Hybrid)", f"${improved_cost:.2f}", "Gemini 2.5 Flash Vision (thinking_budget=0)"],
        ]
        print(tabulate.tabulate(cost_rows, headers=cost_headers, tablefmt="grid"))
        print("\n" + "="*80)

    def _estimate_cost(self, approach: str, num_requests: int) -> float:
        """Estimate cost per requests based on approach"""
        costs = {
            "ocr_llm": 0.006,  # $0.006 per request (Gemini 2.5 Flash)
            "hybrid": 0.006,   # $0.006 per request (Gemini 2.5 Flash Vision)
            "vlm_direct": 0.006  # $0.006 per request (Gemini 2.5 Flash Vision)
        }
        return costs.get(approach, 0.05) * num_requests

    def _save_results(self, output_path: str):
        """Save complete results to JSON"""
        def metrics_to_dict(metrics: EvaluationMetrics) -> dict:
            return {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'step_level_accuracy': metrics.step_level_accuracy,
                'error_detection_rate': metrics.error_detection_rate,
                'false_positive_rate': metrics.false_positive_rate,
                'latency_p50': metrics.latency_p50,
                'latency_p90': metrics.latency_p90,
                'latency_p95': metrics.latency_p95,
            }

        serializable_results = {
            'baseline_metrics': metrics_to_dict(self.results['baseline_metrics']),
            'improved_metrics': metrics_to_dict(self.results['improved_metrics']),
            'improvements': self.results['improvements'],
            'evaluation_metadata': self.results['evaluation_metadata']
        }

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


async def run_evaluation():
    """Run evaluation using the real sample images"""
    print("="*60)
    print("ERROR DETECTION API - REAL DATA EVALUATION")
    print("="*60)

    setup_logging()

    # Create real dataset
    print("\n📂 Creating dataset from real sample images...")
    dataset = create_real_dataset()
    dataset = convert_to_web_urls(dataset)

    # Save the real dataset
    real_dataset_path = "./data/real_eval_dataset.json"
    Path(real_dataset_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_dataset(real_dataset_path)

    # Print dataset summary
    print(f"✅ Created dataset with {len(dataset.samples)} samples")
    print(f"📍 Problems found:")

    for sample in dataset.samples:
        status = "❌ HAS ERROR" if sample.ground_truth_error else "✅ CORRECT"
        topic = sample.metadata.get('topic', 'unknown').upper()
        print(f"  • {sample.id} ({topic}): {status}")
        if sample.ground_truth_error:
            print(f"    └─ {sample.ground_truth_error}")

    # Modified evaluation arguments for real data
    class RealEvalArgs:
        def __init__(self):
            self.dataset_path = real_dataset_path
            self.output = "./data/eval_results/real_evaluation_results.json"
            self.export_cases = "./data/eval_results/real_detailed_results.json"
            self.max_concurrent = 1  # Reduce to 1 to avoid rate limits
            self.seed = 42

    args = RealEvalArgs()

    # Run evaluation harness
    print(f"\n🚀 Starting evaluation with real data...")
    print(f"   Dataset: {args.dataset_path}")
    print(f"   Max concurrent: {args.max_concurrent}")

    harness = EvaluationHarness(args)
    results = await harness.run_evaluation()

    # Additional analysis for real data
    print(f"\n📊 REAL DATA ANALYSIS:")
    print(f"=" * 40)

    baseline_metrics = results['baseline_metrics']
    improved_metrics = results['improved_metrics']

    print(f"Real Sample Performance:")
    print(f"  Baseline Error Detection: {baseline_metrics.error_detection_rate:.1%}")
    print(f"  Improved Error Detection: {improved_metrics.error_detection_rate:.1%}")
    print(f"  Accuracy Improvement: {results['improvements']['accuracy_improvement']:+.1%}")

    # Error type analysis
    print(f"\nError Types in Real Data:")
    error_types = {}
    for sample in dataset.samples:
        if sample.ground_truth_error:
            for label in sample.step_level_labels:
                if label.get('has_error') and label.get('error_type'):
                    error_type = label['error_type']
                    error_types[error_type] = error_types.get(error_type, 0) + 1

    for error_type, count in error_types.items():
        print(f"  • {error_type}: {count} occurrences")

    print(f"\n🎯 Real vs Synthetic Comparison:")
    print(f"   Real data provides more challenging, authentic test cases")
    print(f"   Handwriting variations and realistic student errors")
    print(f"   Better reflection of actual API performance in production")

    return results


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run error detection evaluation")
    parser.add_argument("--dataset-path", default="./data/real_eval_dataset.json",
                       help="Path to evaluation dataset")
    parser.add_argument("--output", default="./data/eval_results/evaluation_results.json",
                       help="Output path for results")
    parser.add_argument("--export-cases", help="Export per-case results to CSV/JSON")
    parser.add_argument("--max-concurrent", type=int, default=2,
                       help="Maximum concurrent requests during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Setup
    setup_logging()

    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    harness = EvaluationHarness(args)
    results = await harness.run_evaluation()
    return results


if __name__ == "__main__":
    try:
        import tabulate
    except ImportError:
        print("Installing required dependency: tabulate")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        import tabulate

    asyncio.run(main())