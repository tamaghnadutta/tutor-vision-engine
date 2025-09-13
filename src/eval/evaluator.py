"""
Evaluation framework for error detection models
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from statistics import median
from dataclasses import asdict

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data.dataset import ErrorDetectionDataset, DatasetSample, EvaluationMetrics
from src.models.error_detector import ErrorDetector

logger = logging.getLogger(__name__)


class ErrorDetectionEvaluator:
    """Evaluator for error detection models"""

    def __init__(self, dataset_path: str):
        self.dataset = ErrorDetectionDataset(dataset_path)
        self.results: List[Dict[str, Any]] = []

    async def evaluate_model(self,
                           model: ErrorDetector,
                           model_name: str = "default",
                           max_concurrent: int = 5) -> EvaluationMetrics:
        """Evaluate a model on the dataset"""
        logger.info(f"Starting evaluation of {model_name}")

        samples = self.dataset.get_samples()
        if not samples:
            logger.warning("No samples found in dataset")
            return self._empty_metrics()

        # Track metrics
        predictions = []
        ground_truths = []
        latencies = []
        step_level_results = []

        # Evaluate with controlled concurrency
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(self._evaluate_sample, model, sample): sample
                for sample in samples
            }

            # Collect results
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    predictions.append(result['prediction'])
                    ground_truths.append(result['ground_truth'])
                    latencies.append(result['latency'])
                    step_level_results.extend(result['step_level_results'])

                    # Store detailed result
                    self.results.append({
                        'sample_id': sample.id,
                        'model_name': model_name,
                        'prediction': result['prediction'],
                        'ground_truth': result['ground_truth'],
                        'latency': result['latency'],
                        'step_level_results': result['step_level_results']
                    })

                except Exception as e:
                    logger.error(f"Error evaluating sample {sample.id}: {e}")
                    # Add failed result
                    predictions.append({'has_error': False, 'error': None})
                    ground_truths.append({'has_error': bool(sample.ground_truth_error), 'error': sample.ground_truth_error})
                    latencies.append(30.0)  # Assume timeout

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, ground_truths, latencies, step_level_results)

        logger.info(f"Evaluation completed for {model_name}: accuracy={metrics.accuracy:.3f}, latency_p95={metrics.latency_p95:.2f}s")

        return metrics

    def _evaluate_sample(self, model: ErrorDetector, sample: DatasetSample) -> Dict[str, Any]:
        """Evaluate a single sample"""
        start_time = time.time()

        try:
            # Run inference
            result = model.detect_errors_sync(
                question_url=sample.question_url,
                solution_url=sample.solution_url,
                bounding_box=None,  # Use full image
                context={'sample_id': sample.id}
            )

            latency = time.time() - start_time

            # Format prediction
            prediction = {
                'has_error': bool(result.get('error')),
                'error': result.get('error'),
                'correction': result.get('correction'),
                'hint': result.get('hint')
            }

            # Format ground truth
            ground_truth = {
                'has_error': bool(sample.ground_truth_error),
                'error': sample.ground_truth_error,
                'correction': sample.ground_truth_correction,
                'hint': sample.ground_truth_hint
            }

            # Evaluate step-level accuracy
            step_level_results = self._evaluate_step_level(
                result.get('solution_lines', []),
                sample.step_level_labels
            )

            return {
                'prediction': prediction,
                'ground_truth': ground_truth,
                'latency': latency,
                'step_level_results': step_level_results
            }

        except Exception as e:
            logger.error(f"Error in sample evaluation: {e}")
            latency = time.time() - start_time
            return {
                'prediction': {'has_error': False, 'error': None},
                'ground_truth': {
                    'has_error': bool(sample.ground_truth_error),
                    'error': sample.ground_truth_error
                },
                'latency': latency,
                'step_level_results': []
            }

    def _evaluate_step_level(self, predicted_steps: List[str],
                           ground_truth_labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate step-level accuracy"""
        results = []

        for i, gt_label in enumerate(ground_truth_labels):
            predicted_error = i < len(predicted_steps) and 'error' in predicted_steps[i].lower()
            actual_error = gt_label.get('has_error', False)

            results.append({
                'step': gt_label.get('step', i),
                'predicted_error': predicted_error,
                'actual_error': actual_error,
                'correct': predicted_error == actual_error
            })

        return results

    def _calculate_metrics(self, predictions: List[Dict], ground_truths: List[Dict],
                         latencies: List[float], step_level_results: List[Dict]) -> EvaluationMetrics:
        """Calculate evaluation metrics"""

        # Error detection metrics
        tp = sum(1 for p, gt in zip(predictions, ground_truths)
                if p['has_error'] and gt['has_error'])
        fp = sum(1 for p, gt in zip(predictions, ground_truths)
                if p['has_error'] and not gt['has_error'])
        fn = sum(1 for p, gt in zip(predictions, ground_truths)
                if not p['has_error'] and gt['has_error'])
        tn = sum(1 for p, gt in zip(predictions, ground_truths)
                if not p['has_error'] and not gt['has_error'])

        # Calculate metrics with safeguards
        accuracy = (tp + tn) / len(predictions) if predictions else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Step-level accuracy
        step_level_accuracy = (
            sum(1 for result in step_level_results if result['correct']) / len(step_level_results)
            if step_level_results else 0.0
        )

        # Error detection rate
        error_detection_rate = tp / sum(1 for gt in ground_truths if gt['has_error']) if ground_truths else 0.0

        # False positive rate
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Latency percentiles
        sorted_latencies = sorted(latencies) if latencies else [0.0]
        latency_p50 = self._percentile(sorted_latencies, 50)
        latency_p90 = self._percentile(sorted_latencies, 90)
        latency_p95 = self._percentile(sorted_latencies, 95)

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            step_level_accuracy=step_level_accuracy,
            error_detection_rate=error_detection_rate,
            false_positive_rate=false_positive_rate,
            latency_p50=latency_p50,
            latency_p90=latency_p90,
            latency_p95=latency_p95
        )

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        index = int((percentile / 100) * len(data))
        return data[min(index, len(data) - 1)]

    def _empty_metrics(self) -> EvaluationMetrics:
        """Return empty metrics for failed evaluation"""
        return EvaluationMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
            step_level_accuracy=0.0, error_detection_rate=0.0,
            false_positive_rate=0.0, latency_p50=0.0, latency_p90=0.0, latency_p95=0.0
        )

    def save_results(self, output_path: str) -> None:
        """Save evaluation results to file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")

    def compare_models(self, baseline_metrics: EvaluationMetrics,
                      improved_metrics: EvaluationMetrics) -> Dict[str, float]:
        """Compare two models and return improvement metrics"""
        improvements = {
            'accuracy_improvement': improved_metrics.accuracy - baseline_metrics.accuracy,
            'f1_improvement': improved_metrics.f1_score - baseline_metrics.f1_score,
            'latency_p95_improvement': baseline_metrics.latency_p95 - improved_metrics.latency_p95,
            'step_level_accuracy_improvement': improved_metrics.step_level_accuracy - baseline_metrics.step_level_accuracy
        }

        return improvements