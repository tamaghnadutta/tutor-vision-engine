#!/usr/bin/env python3
"""
Analytics Dashboard for Error Detection API
Provides comprehensive analytics and visualizations
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.result_storage import result_storage
from src.utils.logging import setup_logging
import logging

logger = logging.getLogger(__name__)

class AnalyticsDashboard:
    """Interactive analytics dashboard"""

    def __init__(self):
        self.storage = result_storage
        setup_logging()

    def generate_performance_charts(self, days_back: int = 30, output_dir: str = "./data/analytics"):
        """Generate performance visualization charts"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report = self.storage.generate_analytics_report(days_back)

        if report.total_runs == 0:
            print(f"No evaluation runs found in the last {days_back} days")
            return

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Accuracy Comparison Chart
        self._create_accuracy_comparison_chart(report, output_path)

        # 2. Cost Analysis Chart
        self._create_cost_analysis_chart(report, output_path)

        # 3. Performance Trends Chart
        self._create_performance_trends_chart(days_back, output_path)

        # 4. Latency vs Accuracy Scatter Plot
        self._create_latency_accuracy_scatter(report, output_path)

        print(f"📊 Generated analytics charts in {output_path}")

    def _create_accuracy_comparison_chart(self, report, output_path):
        """Create accuracy comparison bar chart"""
        if not report.model_comparison:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Accuracy comparison
        approaches = []
        accuracies = []
        f1_scores = []

        for approach in ['ocr_llm', 'vlm_direct', 'hybrid']:
            if ('accuracy', 'mean') in report.model_comparison:
                acc_data = report.model_comparison[('accuracy', 'mean')]
                f1_data = report.model_comparison[('f1_score', 'mean')]

                if approach in acc_data:
                    approaches.append(approach.replace('_', '→'))
                    accuracies.append(acc_data[approach])
                    f1_scores.append(f1_data.get(approach, 0))

        if approaches:
            ax1.bar(approaches, accuracies, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax1.set_title('Average Accuracy by Approach')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)

            # Add value labels on bars
            for i, v in enumerate(accuracies):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

            ax2.bar(approaches, f1_scores, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax2.set_title('Average F1 Score by Approach')
            ax2.set_ylabel('F1 Score')
            ax2.set_ylim(0, 1)

            # Add value labels on bars
            for i, v in enumerate(f1_scores):
                ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_cost_analysis_chart(self, report, output_path):
        """Create cost analysis charts"""
        if not report.cost_analysis or 'by_approach' not in report.cost_analysis:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        cost_data = report.cost_analysis['by_approach']

        # Extract data for plotting
        approaches = []
        avg_costs = []
        total_tokens = []
        avg_api_calls = []

        for approach in ['ocr_llm', 'vlm_direct', 'hybrid']:
            if ('total_cost', 'mean') in cost_data and approach in cost_data[('total_cost', 'mean')]:
                approaches.append(approach.replace('_', '→'))
                avg_costs.append(cost_data[('total_cost', 'mean')][approach])
                total_tokens.append(cost_data[('total_tokens', 'sum')].get(approach, 0))
                avg_api_calls.append(cost_data[('api_calls', 'mean')].get(approach, 0))

        if approaches:
            # Average cost per run
            ax1.bar(approaches, avg_costs, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax1.set_title('Average Cost per Run')
            ax1.set_ylabel('Cost (USD)')
            for i, v in enumerate(avg_costs):
                ax1.text(i, v + max(avg_costs) * 0.01, f'${v:.6f}', ha='center', va='bottom')

            # Total tokens used
            ax2.bar(approaches, total_tokens, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax2.set_title('Total Tokens Used')
            ax2.set_ylabel('Tokens')
            for i, v in enumerate(total_tokens):
                ax2.text(i, v + max(total_tokens) * 0.01, f'{v:,}', ha='center', va='bottom')

            # Average API calls
            ax3.bar(approaches, avg_api_calls, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax3.set_title('Average API Calls per Run')
            ax3.set_ylabel('API Calls')
            for i, v in enumerate(avg_api_calls):
                ax3.text(i, v + max(avg_api_calls) * 0.01, f'{v:.1f}', ha='center', va='bottom')

            # Cost efficiency (accuracy per dollar)
            if avg_costs and ('accuracy', 'mean') in report.model_comparison:
                acc_data = report.model_comparison[('accuracy', 'mean')]
                efficiency = []
                for i, approach in enumerate(['ocr_llm', 'vlm_direct', 'hybrid']):
                    if approach in acc_data and avg_costs[i] > 0:
                        efficiency.append(acc_data[approach] / avg_costs[i])
                    else:
                        efficiency.append(0)

                ax4.bar(approaches, efficiency, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax4.set_title('Cost Efficiency (Accuracy per $)')
                ax4.set_ylabel('Accuracy / Cost')
                for i, v in enumerate(efficiency):
                    if v > 0:
                        ax4.text(i, v + max(efficiency) * 0.01, f'{v:.0f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / 'cost_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_performance_trends_chart(self, days_back, output_path):
        """Create performance trends over time"""
        approaches = ['ocr_llm', 'vlm_direct', 'hybrid']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        for i, approach in enumerate(approaches):
            # Get historical accuracy data
            acc_data = self.storage.get_historical_performance(approach, 'accuracy', days_back)
            latency_data = self.storage.get_historical_performance(approach, 'latency_p95', days_back)

            if acc_data:
                dates, accuracies = zip(*acc_data)
                ax1.plot(dates, accuracies, marker='o', label=approach.replace('_', '→'),
                        color=colors[i], linewidth=2, markersize=6)

            if latency_data:
                dates, latencies = zip(*latency_data)
                ax2.plot(dates, latencies, marker='s', label=approach.replace('_', '→'),
                        color=colors[i], linewidth=2, markersize=6)

        ax1.set_title('Accuracy Trends Over Time')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        ax2.set_title('Latency Trends Over Time')
        ax2.set_ylabel('Latency P95 (seconds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'performance_trends.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_latency_accuracy_scatter(self, report, output_path):
        """Create latency vs accuracy scatter plot"""
        if not report.model_comparison:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        approaches = ['ocr_llm', 'vlm_direct', 'hybrid']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        accuracies = []
        latencies = []
        labels = []
        colors_used = []

        for i, approach in enumerate(approaches):
            if (('accuracy', 'mean') in report.model_comparison and
                ('latency_p95', 'mean') in report.model_comparison):

                acc_data = report.model_comparison[('accuracy', 'mean')]
                lat_data = report.model_comparison[('latency_p95', 'mean')]

                if approach in acc_data and approach in lat_data:
                    accuracies.append(acc_data[approach])
                    latencies.append(lat_data[approach])
                    labels.append(approach.replace('_', '→'))
                    colors_used.append(colors[i])

        if accuracies:
            scatter = ax.scatter(latencies, accuracies, c=colors_used, s=200, alpha=0.8, edgecolors='black')

            # Add labels for each point
            for i, label in enumerate(labels):
                ax.annotate(label, (latencies[i], accuracies[i]), xytext=(10, 10),
                          textcoords='offset points', fontsize=12, fontweight='bold')

            ax.set_xlabel('Latency P95 (seconds)')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy vs Latency Trade-off')
            ax.grid(True, alpha=0.3)

            # Add ideal region annotation
            ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good Accuracy (>0.8)')
            ax.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='Target Latency (<10s)')
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_path / 'latency_accuracy_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_detailed_report(self, days_back: int = 30, output_file: str = "./data/analytics_report.txt"):
        """Generate detailed text report"""
        report = self.storage.generate_analytics_report(days_back)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"ERROR DETECTION API - DETAILED ANALYTICS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: Last {days_back} days\n\n")

            # Overview
            f.write("📊 OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total evaluation runs: {report.total_runs}\n")
            f.write(f"Date range: {report.date_range[0].strftime('%Y-%m-%d')} to {report.date_range[1].strftime('%Y-%m-%d')}\n\n")

            # Best performance
            if report.best_accuracy:
                f.write("🏆 BEST PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                f.write(f"Best run: {report.best_accuracy['run_id']}\n")
                f.write(f"Approach: {report.best_accuracy['approach']}\n")
                f.write(f"Accuracy: {report.best_accuracy['accuracy']:.4f}\n")
                f.write(f"F1 Score: {report.best_accuracy['f1_score']:.4f}\n")
                f.write(f"Timestamp: {report.best_accuracy['timestamp']}\n\n")

            # Cost analysis
            if report.cost_analysis:
                f.write("💰 COST ANALYSIS\n")
                f.write("-" * 40 + "\n")
                cost = report.cost_analysis
                f.write(f"Total spent: ${cost['total_spent']:.6f}\n")
                f.write(f"Total tokens: {cost['total_tokens']:,}\n")
                f.write(f"Total API calls: {cost['total_api_calls']}\n")
                f.write(f"Average cost per run: ${cost['avg_cost_per_run']:.6f}\n\n")

                if 'by_approach' in cost:
                    f.write("Cost breakdown by approach:\n")
                    for approach in ['ocr_llm', 'vlm_direct', 'hybrid']:
                        if ('total_cost', 'mean') in cost['by_approach']:
                            mean_cost = cost['by_approach'][('total_cost', 'mean')].get(approach, 0)
                            total_cost = cost['by_approach'][('total_cost', 'sum')].get(approach, 0)
                            f.write(f"  {approach.replace('_', '→')}: avg=${mean_cost:.6f}, total=${total_cost:.6f}\n")
                    f.write("\n")

            # Model comparison
            if report.model_comparison:
                f.write("⚖️ MODEL COMPARISON\n")
                f.write("-" * 40 + "\n")

                for approach in ['ocr_llm', 'vlm_direct', 'hybrid']:
                    f.write(f"\n{approach.replace('_', '→').upper()}:\n")

                    if ('accuracy', 'mean') in report.model_comparison:
                        acc = report.model_comparison[('accuracy', 'mean')].get(approach, 0)
                        acc_std = report.model_comparison[('accuracy', 'std')].get(approach, 0)
                        f.write(f"  Accuracy: {acc:.4f} ± {acc_std:.4f}\n")

                    if ('f1_score', 'mean') in report.model_comparison:
                        f1 = report.model_comparison[('f1_score', 'mean')].get(approach, 0)
                        f1_std = report.model_comparison[('f1_score', 'std')].get(approach, 0)
                        f.write(f"  F1 Score: {f1:.4f} ± {f1_std:.4f}\n")

                    if ('latency_p95', 'mean') in report.model_comparison:
                        lat = report.model_comparison[('latency_p95', 'mean')].get(approach, 0)
                        lat_std = report.model_comparison[('latency_p95', 'std')].get(approach, 0)
                        f.write(f"  Latency P95: {lat:.2f} ± {lat_std:.2f} seconds\n")

            f.write("\n" + "="*80 + "\n")

        print(f"📝 Generated detailed report: {output_path}")

    def clean_old_data(self, days_to_keep: int = 90):
        """Clean old data and cache entries"""
        deleted_cache = self.storage.clean_expired_cache()

        # Could add logic to archive/delete old evaluation runs here
        print(f"🧹 Cleaned {deleted_cache} expired cache entries")
        print(f"📅 Data retention policy: {days_to_keep} days")

def main():
    parser = argparse.ArgumentParser(description="Analytics Dashboard for Error Detection API")
    parser.add_argument("--days", type=int, default=7, help="Number of days to analyze")
    parser.add_argument("--charts", action="store_true", help="Generate visualization charts")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("--summary", action="store_true", help="Print analytics summary")
    parser.add_argument("--export", action="store_true", help="Export data to CSV")
    parser.add_argument("--clean", action="store_true", help="Clean old data and cache")

    args = parser.parse_args()

    dashboard = AnalyticsDashboard()

    if args.summary:
        dashboard.storage.print_analytics_summary(args.days)

    if args.charts:
        dashboard.generate_performance_charts(args.days)

    if args.report:
        dashboard.generate_detailed_report(args.days)

    if args.export:
        dashboard.storage.export_to_csv()

    if args.clean:
        dashboard.clean_old_data()

    # If no specific action, show summary
    if not any([args.charts, args.report, args.export, args.clean]):
        dashboard.storage.print_analytics_summary(args.days)

if __name__ == "__main__":
    main()