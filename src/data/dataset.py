"""
Dataset handling for error detection evaluation
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    """Single dataset sample for error detection"""
    id: str
    question_url: str
    solution_url: str
    ground_truth_error: Optional[str]
    ground_truth_correction: Optional[str]
    ground_truth_hint: Optional[str]
    step_level_labels: List[Dict[str, Any]]
    is_noisy: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for error detection"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    step_level_accuracy: float
    error_detection_rate: float
    false_positive_rate: float
    latency_p50: float
    latency_p90: float
    latency_p95: float


class ErrorDetectionDataset:
    """Dataset class for error detection evaluation"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.samples: List[DatasetSample] = []
        self.load_dataset()

    def load_dataset(self) -> None:
        """Load dataset from JSON file"""
        if not self.dataset_path.exists():
            logger.warning(f"Dataset file not found: {self.dataset_path}")
            return

        with open(self.dataset_path, 'r') as f:
            data = json.load(f)

        for item in data.get('samples', []):
            sample = DatasetSample(
                id=item['id'],
                question_url=item['question_url'],
                solution_url=item['solution_url'],
                ground_truth_error=item.get('ground_truth_error'),
                ground_truth_correction=item.get('ground_truth_correction'),
                ground_truth_hint=item.get('ground_truth_hint'),
                step_level_labels=item.get('step_level_labels', []),
                is_noisy=item.get('is_noisy', False),
                metadata=item.get('metadata', {})
            )
            self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} samples from dataset")

    def get_samples(self, noisy_only: bool = False) -> List[DatasetSample]:
        """Get dataset samples, optionally filtered by noisy samples"""
        if noisy_only:
            return [sample for sample in self.samples if sample.is_noisy]
        return self.samples

    def get_train_test_split(self, test_ratio: float = 0.2) -> Tuple[List[DatasetSample], List[DatasetSample]]:
        """Split dataset into train/test sets"""
        split_index = int(len(self.samples) * (1 - test_ratio))
        return self.samples[:split_index], self.samples[split_index:]

    def save_dataset(self, output_path: str) -> None:
        """Save dataset to JSON file"""
        data = {
            'metadata': {
                'total_samples': len(self.samples),
                'noisy_samples': len([s for s in self.samples if s.is_noisy]),
                'total_step_lines': sum(len(s.step_level_labels) for s in self.samples)
            },
            'samples': []
        }

        for sample in self.samples:
            data['samples'].append({
                'id': sample.id,
                'question_url': sample.question_url,
                'solution_url': sample.solution_url,
                'ground_truth_error': sample.ground_truth_error,
                'ground_truth_correction': sample.ground_truth_correction,
                'ground_truth_hint': sample.ground_truth_hint,
                'step_level_labels': sample.step_level_labels,
                'is_noisy': sample.is_noisy,
                'metadata': sample.metadata or {}
            })

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved dataset with {len(self.samples)} samples to {output_path}")


