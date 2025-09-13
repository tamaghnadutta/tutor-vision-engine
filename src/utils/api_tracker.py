"""
API Call Tracker for capturing token usage and costs
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import threading
import logging
import time
from src.utils.cost_calculator import CostCalculator, TokenUsage, CostBreakdown

logger = logging.getLogger(__name__)

@dataclass
class APICall:
    """Record of a single API call"""
    model: str
    provider: str
    purpose: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    duration: float
    timestamp: float

@dataclass
class SessionTracker:
    """Tracks API calls for a session"""
    calls: List[APICall] = field(default_factory=list)
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0

    def add_call(self, call: APICall):
        self.calls.append(call)
        self.total_cost += call.cost
        self.total_input_tokens += call.input_tokens
        self.total_output_tokens += call.output_tokens
        self.total_calls += 1

class APITracker:
    """Global API call tracker"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.cost_calculator = CostCalculator()
                    cls._instance.session = SessionTracker()
        return cls._instance

    def track_openai_call(self, completion, model: str, purpose: str, duration: float) -> APICall:
        """Track an OpenAI API call"""
        usage = completion.usage

        token_usage = TokenUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )

        cost_breakdown = self.cost_calculator.calculate_cost(model, token_usage)

        call = APICall(
            model=model,
            provider="OpenAI",
            purpose=purpose,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=cost_breakdown.total_cost,
            duration=duration,
            timestamp=time.time()
        )

        self.session.add_call(call)

        logger.info(f"API Call [{purpose}] - Model: {model}, Tokens: {usage.total_tokens}, "
                   f"Cost: ${cost_breakdown.total_cost:.6f}, Duration: {duration:.2f}s")

        return call

    def track_gemini_call(self, response, model: str, purpose: str, duration: float,
                         estimated_input_tokens: int = 1000, estimated_output_tokens: int = 200) -> APICall:
        """Track a Gemini API call (estimated tokens since Gemini doesn't return usage)"""

        # Gemini doesn't provide token usage, so we estimate based on content
        # In a real implementation, you'd want to use a tokenizer to count more accurately
        input_tokens = estimated_input_tokens
        output_tokens = len(response.text.split()) * 1.3 if hasattr(response, 'text') else estimated_output_tokens

        token_usage = TokenUsage(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            total_tokens=int(input_tokens + output_tokens)
        )

        cost_breakdown = self.cost_calculator.calculate_cost(model, token_usage)

        call = APICall(
            model=model,
            provider="Google",
            purpose=purpose,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            total_tokens=int(input_tokens + output_tokens),
            cost=cost_breakdown.total_cost,
            duration=duration,
            timestamp=time.time()
        )

        self.session.add_call(call)

        logger.info(f"API Call [{purpose}] - Model: {model}, Tokens: ~{int(input_tokens + output_tokens)}, "
                   f"Cost: ~${cost_breakdown.total_cost:.6f}, Duration: {duration:.2f}s")

        return call

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        if not self.session.calls:
            return {
                "total_calls": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "calls": []
            }

        # Group by purpose
        by_purpose = {}
        for call in self.session.calls:
            if call.purpose not in by_purpose:
                by_purpose[call.purpose] = {
                    "calls": 0,
                    "cost": 0.0,
                    "tokens": 0,
                    "models": set()
                }

            by_purpose[call.purpose]["calls"] += 1
            by_purpose[call.purpose]["cost"] += call.cost
            by_purpose[call.purpose]["tokens"] += call.total_tokens
            by_purpose[call.purpose]["models"].add(call.model)

        # Convert sets to lists for JSON serialization
        for purpose_data in by_purpose.values():
            purpose_data["models"] = list(purpose_data["models"])

        return {
            "total_calls": self.session.total_calls,
            "total_cost": self.session.total_cost,
            "total_input_tokens": self.session.total_input_tokens,
            "total_output_tokens": self.session.total_output_tokens,
            "total_tokens": self.session.total_input_tokens + self.session.total_output_tokens,
            "by_purpose": by_purpose,
            "calls": [
                {
                    "model": call.model,
                    "provider": call.provider,
                    "purpose": call.purpose,
                    "tokens": call.total_tokens,
                    "cost": call.cost,
                    "duration": call.duration
                }
                for call in self.session.calls
            ]
        }

    def reset_session(self):
        """Reset the current session"""
        self.session = SessionTracker()

    def print_session_summary(self):
        """Print a detailed session summary"""
        summary = self.get_session_summary()

        if summary["total_calls"] == 0:
            print("No API calls tracked in this session.")
            return

        print(f"\n{'='*60}")
        print(f"API CALL SUMMARY - SESSION REPORT")
        print(f"{'='*60}")

        print(f"📊 Overall Statistics:")
        print(f"   Total API calls: {summary['total_calls']}")
        print(f"   Total tokens: {summary['total_tokens']:,}")
        print(f"   Total cost: ${summary['total_cost']:.6f}")

        print(f"\n📈 Token Breakdown:")
        print(f"   Input tokens: {summary['total_input_tokens']:,}")
        print(f"   Output tokens: {summary['total_output_tokens']:,}")
        print(f"   Ratio (out/in): {summary['total_output_tokens']/max(summary['total_input_tokens'], 1):.2f}")

        print(f"\n🔍 By Purpose:")
        for purpose, data in summary["by_purpose"].items():
            print(f"   {purpose}:")
            print(f"     Calls: {data['calls']}")
            print(f"     Tokens: {data['tokens']:,}")
            print(f"     Cost: ${data['cost']:.6f}")
            print(f"     Models: {', '.join(data['models'])}")


# Global tracker instance
api_tracker = APITracker()

def track_api_call(completion_or_response, model: str, purpose: str, duration: float,
                   provider: str = "openai", **kwargs) -> APICall:
    """Convenience function to track API calls"""
    if provider.lower() == "openai":
        return api_tracker.track_openai_call(completion_or_response, model, purpose, duration)
    elif provider.lower() == "google" or provider.lower() == "gemini":
        return api_tracker.track_gemini_call(
            completion_or_response, model, purpose, duration,
            kwargs.get("estimated_input_tokens", 1000),
            kwargs.get("estimated_output_tokens", 200)
        )
    else:
        logger.warning(f"Unknown provider: {provider}")
        return None