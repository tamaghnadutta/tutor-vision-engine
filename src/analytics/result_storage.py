"""
Advanced Result Storage and Analytics System
Handles caching, historical analysis, and performance tracking
"""

import json
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import pickle
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class EvaluationRun:
    """Single evaluation run record"""
    run_id: str
    timestamp: datetime
    approach: str
    dataset_hash: str
    config_hash: str
    accuracy: float
    f1_score: float
    latency_p95: float
    total_cost: float
    total_tokens: int
    api_calls: int
    sample_count: int
    metadata: Dict[str, Any]

@dataclass
class AnalyticsReport:
    """Analytics report for evaluation runs"""
    total_runs: int
    date_range: Tuple[datetime, datetime]
    best_accuracy: Dict[str, Any]
    cost_analysis: Dict[str, Any]
    performance_trends: Dict[str, Any]
    model_comparison: Dict[str, Any]

class ResultStorage:
    """Advanced result storage with SQLite backend and caching"""

    def __init__(self, db_path: str = "./data/analytics.db", cache_dir: str = "./data/cache"):
        self.db_path = Path(db_path)
        self.cache_dir = Path(cache_dir)

        # Create directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    approach TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    latency_p50 REAL NOT NULL,
                    latency_p90 REAL NOT NULL,
                    latency_p95 REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    api_calls INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sample_results (
                    result_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    ground_truth_error TEXT,
                    predicted_error TEXT,
                    confidence REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    cost REAL NOT NULL,
                    tokens_used INTEGER NOT NULL,
                    correct_prediction INTEGER NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs (run_id)
                );

                CREATE TABLE IF NOT EXISTS cached_results (
                    cache_key TEXT PRIMARY KEY,
                    approach TEXT NOT NULL,
                    sample_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_run_timestamp ON evaluation_runs(timestamp);
                CREATE INDEX IF NOT EXISTS idx_run_approach ON evaluation_runs(approach);
                CREATE INDEX IF NOT EXISTS idx_sample_run ON sample_results(run_id);
                CREATE INDEX IF NOT EXISTS idx_cache_key ON cached_results(cache_key);
                CREATE INDEX IF NOT EXISTS idx_cache_expires ON cached_results(expires_at);
            ''')

    def store_evaluation_run(self, run_id: str, approach: str, metrics: Dict[str, Any],
                           api_usage: Dict[str, Any], sample_results: List[Dict[str, Any]],
                           dataset_info: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Store complete evaluation run with all details"""

        # Generate hashes for dataset and config
        dataset_hash = self._hash_dict(dataset_info)
        config_hash = self._hash_dict(config)

        timestamp = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Store main evaluation run
            conn.execute('''
                INSERT OR REPLACE INTO evaluation_runs
                (run_id, timestamp, approach, dataset_hash, config_hash, accuracy, f1_score,
                 precision_score, recall_score, latency_p50, latency_p90, latency_p95,
                 total_cost, total_tokens, api_calls, sample_count, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id, timestamp, approach, dataset_hash, config_hash,
                metrics.get('accuracy', 0.0), metrics.get('f1_score', 0.0),
                metrics.get('precision', 0.0), metrics.get('recall', 0.0),
                metrics.get('latency_p50', 0.0), metrics.get('latency_p90', 0.0), metrics.get('latency_p95', 0.0),
                api_usage.get('total_cost', 0.0), api_usage.get('total_tokens', 0),
                api_usage.get('total_calls', 0), len(sample_results),
                json.dumps({
                    'dataset_info': dataset_info,
                    'config': config,
                    'api_breakdown': api_usage.get('by_purpose', {}),
                    'detailed_metrics': metrics
                })
            ))

            # Store individual sample results
            for sample_result in sample_results:
                result_id = f"{run_id}_{sample_result.get('sample_id', 'unknown')}"
                conn.execute('''
                    INSERT OR REPLACE INTO sample_results
                    (result_id, run_id, sample_id, ground_truth_error, predicted_error,
                     confidence, processing_time, cost, tokens_used, correct_prediction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_id, run_id, sample_result.get('sample_id'),
                    sample_result.get('ground_truth_error'),
                    sample_result.get('predicted_error'),
                    sample_result.get('confidence', 0.0),
                    sample_result.get('processing_time', 0.0),
                    sample_result.get('cost', 0.0),
                    sample_result.get('tokens_used', 0),
                    1 if sample_result.get('correct_prediction', False) else 0
                ))

        logger.info(f"Stored evaluation run {run_id} with {len(sample_results)} sample results")
        return run_id

    def get_cached_result(self, approach: str, sample_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result for a specific sample and approach"""
        cache_key = f"{approach}_{sample_hash}"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT result_json, expires_at FROM cached_results
                WHERE cache_key = ? AND expires_at > ?
            ''', (cache_key, datetime.now().isoformat()))

            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    logger.warning(f"Invalid cached result for {cache_key}")

        return None

    def cache_result(self, approach: str, sample_hash: str, result: Dict[str, Any],
                    expires_hours: int = 24):
        """Cache a result for future use"""
        cache_key = f"{approach}_{sample_hash}"
        timestamp = datetime.now()
        expires_at = timestamp + timedelta(hours=expires_hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO cached_results
                (cache_key, approach, sample_hash, result_json, timestamp, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                cache_key, approach, sample_hash, json.dumps(result),
                timestamp.isoformat(), expires_at.isoformat()
            ))

        logger.debug(f"Cached result for {cache_key}")

    def clean_expired_cache(self) -> int:
        """Clean expired cache entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                DELETE FROM cached_results WHERE expires_at <= ?
            ''', (datetime.now().isoformat(),))

            deleted = cursor.rowcount
            logger.info(f"Cleaned {deleted} expired cache entries")
            return deleted

    def generate_analytics_report(self, days_back: int = 30) -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        start_date = datetime.now() - timedelta(days=days_back)

        with sqlite3.connect(self.db_path) as conn:
            # Get evaluation runs
            df_runs = pd.read_sql_query('''
                SELECT * FROM evaluation_runs
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', conn, params=(start_date.isoformat(),))

            if df_runs.empty:
                return AnalyticsReport(
                    total_runs=0,
                    date_range=(start_date, datetime.now()),
                    best_accuracy={},
                    cost_analysis={},
                    performance_trends={},
                    model_comparison={}
                )

            # Get sample results
            df_samples = pd.read_sql_query('''
                SELECT sr.* FROM sample_results sr
                JOIN evaluation_runs er ON sr.run_id = er.run_id
                WHERE er.timestamp >= ?
            ''', conn, params=(start_date.isoformat(),))

        # Generate analytics
        analytics = self._analyze_dataframes(df_runs, df_samples)

        return AnalyticsReport(
            total_runs=len(df_runs),
            date_range=(start_date, datetime.now()),
            **analytics
        )

    def _analyze_dataframes(self, df_runs: pd.DataFrame, df_samples: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataframes to generate insights"""

        # Best accuracy analysis
        best_accuracy = {}
        if not df_runs.empty:
            best_run = df_runs.loc[df_runs['accuracy'].idxmax()]
            best_accuracy = {
                'approach': best_run['approach'],
                'accuracy': best_run['accuracy'],
                'f1_score': best_run['f1_score'],
                'run_id': best_run['run_id'],
                'timestamp': best_run['timestamp']
            }

        # Cost analysis
        cost_analysis = {}
        if not df_runs.empty:
            cost_stats = df_runs.groupby('approach').agg({
                'total_cost': ['mean', 'min', 'max', 'sum'],
                'total_tokens': ['mean', 'sum'],
                'api_calls': ['mean', 'sum']
            }).round(6)

            cost_analysis = {
                'by_approach': cost_stats.to_dict(),
                'total_spent': df_runs['total_cost'].sum(),
                'total_tokens': df_runs['total_tokens'].sum(),
                'total_api_calls': df_runs['api_calls'].sum(),
                'avg_cost_per_run': df_runs['total_cost'].mean()
            }

        # Performance trends
        performance_trends = {}
        if not df_runs.empty:
            df_runs['timestamp'] = pd.to_datetime(df_runs['timestamp'])
            trends = df_runs.groupby(['approach', df_runs['timestamp'].dt.date]).agg({
                'accuracy': 'mean',
                'latency_p95': 'mean',
                'total_cost': 'mean'
            })
            performance_trends = {
                'accuracy_trend': trends['accuracy'].to_dict(),
                'latency_trend': trends['latency_p95'].to_dict(),
                'cost_trend': trends['total_cost'].to_dict()
            }

        # Model comparison
        model_comparison = {}
        if not df_runs.empty:
            comparison = df_runs.groupby('approach').agg({
                'accuracy': ['mean', 'std', 'max'],
                'f1_score': ['mean', 'std', 'max'],
                'latency_p95': ['mean', 'std', 'min'],
                'total_cost': ['mean', 'std', 'min']
            }).round(6)

            model_comparison = comparison.to_dict()

        return {
            'best_accuracy': best_accuracy,
            'cost_analysis': cost_analysis,
            'performance_trends': performance_trends,
            'model_comparison': model_comparison
        }

    def get_historical_performance(self, approach: str, metric: str = 'accuracy',
                                 days_back: int = 30) -> List[Tuple[datetime, float]]:
        """Get historical performance for an approach"""
        start_date = datetime.now() - timedelta(days=days_back)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f'''
                SELECT timestamp, {metric} FROM evaluation_runs
                WHERE approach = ? AND timestamp >= ?
                ORDER BY timestamp
            ''', (approach, start_date.isoformat()))

            return [(datetime.fromisoformat(row[0]), row[1]) for row in cursor.fetchall()]

    def export_to_csv(self, output_dir: str = "./data/exports"):
        """Export all data to CSV files for external analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Export evaluation runs
            df_runs = pd.read_sql_query('SELECT * FROM evaluation_runs', conn)
            df_runs.to_csv(output_path / 'evaluation_runs.csv', index=False)

            # Export sample results
            df_samples = pd.read_sql_query('SELECT * FROM sample_results', conn)
            df_samples.to_csv(output_path / 'sample_results.csv', index=False)

            # Export cache info (without actual results for size)
            df_cache = pd.read_sql_query('''
                SELECT cache_key, approach, timestamp, expires_at
                FROM cached_results
            ''', conn)
            df_cache.to_csv(output_path / 'cache_info.csv', index=False)

        logger.info(f"Exported data to CSV files in {output_path}")

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Generate hash for a dictionary"""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def print_analytics_summary(self, days_back: int = 7):
        """Print a summary of recent analytics"""
        report = self.generate_analytics_report(days_back)

        print(f"\n{'='*80}")
        print(f"ANALYTICS SUMMARY - LAST {days_back} DAYS")
        print(f"{'='*80}")

        print(f"📊 Overview:")
        print(f"   Total evaluation runs: {report.total_runs}")
        print(f"   Date range: {report.date_range[0].strftime('%Y-%m-%d')} to {report.date_range[1].strftime('%Y-%m-%d')}")

        if report.best_accuracy:
            print(f"\n🏆 Best Performance:")
            print(f"   Approach: {report.best_accuracy['approach']}")
            print(f"   Accuracy: {report.best_accuracy['accuracy']:.3f}")
            print(f"   F1 Score: {report.best_accuracy['f1_score']:.3f}")
            print(f"   Run ID: {report.best_accuracy['run_id']}")

        if report.cost_analysis:
            cost = report.cost_analysis
            print(f"\n💰 Cost Analysis:")
            print(f"   Total spent: ${cost['total_spent']:.6f}")
            print(f"   Total tokens: {cost['total_tokens']:,}")
            print(f"   Total API calls: {cost['total_api_calls']}")
            print(f"   Avg cost per run: ${cost['avg_cost_per_run']:.6f}")

        if report.model_comparison:
            print(f"\n⚖️ Model Comparison (Average Performance):")
            for approach in ['ocr_llm', 'vlm_direct', 'hybrid']:
                if ('accuracy', 'mean') in report.model_comparison and approach in [k for k in report.model_comparison[('accuracy', 'mean')].keys()]:
                    acc = report.model_comparison[('accuracy', 'mean')].get(approach, 0)
                    lat = report.model_comparison[('latency_p95', 'mean')].get(approach, 0)
                    cost = report.model_comparison[('total_cost', 'mean')].get(approach, 0)
                    print(f"   {approach.replace('_', '→')}: Acc={acc:.3f}, Latency={lat:.1f}s, Cost=${cost:.6f}")


# Global storage instance
result_storage = ResultStorage()