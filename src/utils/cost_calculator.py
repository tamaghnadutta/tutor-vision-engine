"""
Advanced Cost Calculator for Error Detection API
Based on 2025 pricing for GPT-4o and Gemini-2.5-Flash
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelPricing:
    """Pricing information for a specific model"""
    input_per_million: float  # USD per 1 million input tokens
    output_per_million: float # USD per 1 million output tokens
    name: str
    provider: str

@dataclass
class TokenUsage:
    """Token usage information from API call"""
    input_tokens: int
    output_tokens: int
    total_tokens: int

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for an API call"""
    input_cost: float
    output_cost: float
    total_cost: float
    model: str
    input_tokens: int
    output_tokens: int

# Current 2025 Pricing (as of September 2025)
MODEL_PRICING = {
    # OpenAI GPT-4o - Most commonly used for vision tasks
    "gpt-4o": ModelPricing(
        input_per_million=2.50,   # $2.50 per 1M input tokens
        output_per_million=10.00, # $10.00 per 1M output tokens
        name="GPT-4o",
        provider="OpenAI"
    ),

    # Gemini 2.5 Flash - Current production pricing
    "gemini-2.5-flash": ModelPricing(
        input_per_million=0.30,   # $0.30 per 1M input tokens (4x increase from preview)
        output_per_million=2.50,  # $2.50 per 1M output tokens (4x increase from preview)
        name="Gemini 2.5 Flash",
        provider="Google"
    ),

    # Gemini 2.5 Flash-Lite - More affordable alternative
    "gemini-2.5-flash-lite": ModelPricing(
        input_per_million=0.10,   # $0.10 per 1M input tokens
        output_per_million=0.40,  # $0.40 per 1M output tokens
        name="Gemini 2.5 Flash-Lite",
        provider="Google"
    ),
}

class CostCalculator:
    """Advanced cost calculator with accurate 2025 pricing"""

    def __init__(self):
        self.pricing = MODEL_PRICING

    def calculate_cost(self, model_name: str, usage: TokenUsage) -> CostBreakdown:
        """Calculate cost for a specific API call"""
        model_key = self._normalize_model_name(model_name)
        pricing = self.pricing.get(model_key)

        if not pricing:
            logger.warning(f"No pricing info for model {model_name}, using default")
            pricing = self.pricing["gpt-4o"]  # Default fallback

        # Calculate costs
        input_cost = (usage.input_tokens / 1_000_000) * pricing.input_per_million
        output_cost = (usage.output_tokens / 1_000_000) * pricing.output_per_million
        total_cost = input_cost + output_cost

        return CostBreakdown(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            model=pricing.name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens
        )

    def estimate_approach_cost(self, approach: str, num_requests: int = 100) -> Dict[str, Any]:
        """Estimate cost for different error detection approaches"""

        # Estimated token usage per request based on typical image analysis
        estimates = {
            "ocr_llm": {
                "description": "GPT-4o OCR + GPT-4o reasoning (2 API calls)",
                "models": [
                    {
                        "name": "gpt-4o",
                        "usage": TokenUsage(input_tokens=1500, output_tokens=200, total_tokens=1700),  # OCR step
                        "purpose": "OCR extraction"
                    },
                    {
                        "name": "gpt-4o",
                        "usage": TokenUsage(input_tokens=800, output_tokens=300, total_tokens=1100),   # Reasoning step
                        "purpose": "Text reasoning"
                    }
                ]
            },

            "vlm_direct": {
                "description": "Single GPT-4o vision call",
                "models": [
                    {
                        "name": "gpt-4o",
                        "usage": TokenUsage(input_tokens=2000, output_tokens=400, total_tokens=2400),  # Single call
                        "purpose": "Direct vision analysis"
                    }
                ]
            },

            "vlm_direct_gemini": {
                "description": "Single Gemini 2.5 Flash call",
                "models": [
                    {
                        "name": "gemini-2.5-flash",
                        "usage": TokenUsage(input_tokens=2000, output_tokens=400, total_tokens=2400),
                        "purpose": "Direct vision analysis"
                    }
                ]
            },

            "hybrid": {
                "description": "OCR→LLM + Direct VLM ensemble (3 API calls)",
                "models": [
                    {
                        "name": "gpt-4o",
                        "usage": TokenUsage(input_tokens=1500, output_tokens=200, total_tokens=1700),  # OCR
                        "purpose": "OCR extraction"
                    },
                    {
                        "name": "gpt-4o",
                        "usage": TokenUsage(input_tokens=800, output_tokens=300, total_tokens=1100),   # Reasoning
                        "purpose": "Text reasoning"
                    },
                    {
                        "name": "gpt-4o",
                        "usage": TokenUsage(input_tokens=2000, output_tokens=400, total_tokens=2400), # Direct VLM
                        "purpose": "Direct vision analysis"
                    }
                ]
            }
        }

        if approach not in estimates:
            raise ValueError(f"Unknown approach: {approach}")

        approach_info = estimates[approach]
        total_cost_per_request = 0.0
        cost_breakdown = []

        for model_info in approach_info["models"]:
            cost = self.calculate_cost(model_info["name"], model_info["usage"])
            total_cost_per_request += cost.total_cost
            cost_breakdown.append({
                "model": cost.model,
                "purpose": model_info["purpose"],
                "cost_per_request": cost.total_cost,
                "input_tokens": cost.input_tokens,
                "output_tokens": cost.output_tokens
            })

        return {
            "approach": approach,
            "description": approach_info["description"],
            "cost_per_request": total_cost_per_request,
            "cost_per_100_requests": total_cost_per_request * num_requests,
            "breakdown": cost_breakdown,
            "num_api_calls": len(approach_info["models"])
        }

    def compare_approach_costs(self, num_requests: int = 100) -> Dict[str, Any]:
        """Compare costs across all approaches"""
        approaches = ["ocr_llm", "vlm_direct", "vlm_direct_gemini", "hybrid"]

        comparison = {
            "num_requests": num_requests,
            "approaches": {},
            "summary": {
                "cheapest": {"approach": "", "cost": float('inf')},
                "most_expensive": {"approach": "", "cost": 0.0}
            }
        }

        for approach in approaches:
            try:
                cost_info = self.estimate_approach_cost(approach, num_requests)
                comparison["approaches"][approach] = cost_info

                total_cost = cost_info["cost_per_100_requests"]

                # Track cheapest and most expensive
                if total_cost < comparison["summary"]["cheapest"]["cost"]:
                    comparison["summary"]["cheapest"] = {
                        "approach": approach,
                        "cost": total_cost
                    }

                if total_cost > comparison["summary"]["most_expensive"]["cost"]:
                    comparison["summary"]["most_expensive"] = {
                        "approach": approach,
                        "cost": total_cost
                    }

            except Exception as e:
                logger.error(f"Failed to calculate cost for {approach}: {e}")

        return comparison

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model names to match pricing keys"""
        name_mapping = {
            "gpt-4o": "gpt-4o",
            "gpt-4-vision-preview": "gpt-4o",  # Map older vision model to gpt-4o
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-flash-lite": "gemini-2.5-flash-lite"
        }

        return name_mapping.get(model_name.lower(), model_name.lower())

    def print_cost_report(self, num_requests: int = 100):
        """Print a detailed cost comparison report"""
        comparison = self.compare_approach_costs(num_requests)

        print(f"\n{'='*80}")
        print(f"COST ANALYSIS REPORT - {num_requests} REQUESTS")
        print(f"{'='*80}")
        print(f"Based on current 2025 pricing (September 2025)")

        for approach, info in comparison["approaches"].items():
            print(f"\n📊 {approach.upper().replace('_', ' → ')}")
            print(f"   Description: {info['description']}")
            print(f"   Cost per request: ${info['cost_per_request']:.6f}")
            print(f"   Cost per {num_requests} requests: ${info['cost_per_100_requests']:.2f}")
            print(f"   API calls per request: {info['num_api_calls']}")

            print("   Breakdown:")
            for breakdown in info["breakdown"]:
                print(f"     • {breakdown['model']} ({breakdown['purpose']}): "
                      f"${breakdown['cost_per_request']:.6f}")

        print(f"\n💡 COST INSIGHTS:")
        cheapest = comparison["summary"]["cheapest"]
        most_expensive = comparison["summary"]["most_expensive"]

        print(f"   • Cheapest: {cheapest['approach'].replace('_', ' → ')} "
              f"(${cheapest['cost']:.2f} per {num_requests})")
        print(f"   • Most Expensive: {most_expensive['approach'].replace('_', ' → ')} "
              f"(${most_expensive['cost']:.2f} per {num_requests})")

        savings = most_expensive['cost'] - cheapest['cost']
        print(f"   • Potential Savings: ${savings:.2f} per {num_requests} requests")

        print(f"\n💰 PRICING NOTES:")
        print(f"   • GPT-4o: Output tokens cost 4x more than input tokens")
        print(f"   • Gemini 2.5 Flash: 4x price increase from preview pricing")
        print(f"   • Consider Gemini 2.5 Flash-Lite for cost optimization")
        print(f"   • Batch API can reduce costs by 50% for both models")


if __name__ == "__main__":
    # Example usage and testing
    calculator = CostCalculator()

    # Print detailed cost report
    calculator.print_cost_report(100)

    # Example individual calculation
    usage = TokenUsage(input_tokens=2000, output_tokens=400, total_tokens=2400)
    cost = calculator.calculate_cost("gpt-4o", usage)
    print(f"\n🔍 Example GPT-4o Call:")
    print(f"   Input cost: ${cost.input_cost:.6f}")
    print(f"   Output cost: ${cost.output_cost:.6f}")
    print(f"   Total cost: ${cost.total_cost:.6f}")