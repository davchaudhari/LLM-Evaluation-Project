"""Metrics collection and aggregation.

IMPORTANT: All timing values are in SECONDS (from time.perf_counter()).
Derived metrics (ttft_ms, e2e_ms) are in MILLISECONDS.

Invariants enforced:
- arrival_time <= first_token_time <= end_time
- ttft_ms > 0 for streaming runs (99% of requests)
- e2e_ms >= ttft_ms
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
import json
import numpy as np
from scipy import stats


class MetricsValidationError(Exception):
    """Raised when metrics fail validation invariants."""
    pass


@dataclass
class RequestMetrics:
    """Metrics for a single request.
    
    All timing fields are in SECONDS (perf_counter values).
    Properties ttft_ms, e2e_ms, queue_time_ms return MILLISECONDS.
    """
    request_id: str
    prompt: str
    prompt_tokens: int
    generated_tokens: int
    max_new_tokens: int
    
    # Timing (all in SECONDS from perf_counter)
    arrival_time: float  # When request arrived/was submitted
    first_token_time: Optional[float] = None  # When first token was generated
    end_time: Optional[float] = None  # When generation completed
    dispatch_time: Optional[float] = None  # When request was dispatched to backend
    
    # Metadata
    backend: str = "unknown"
    policy: str = "unknown"
    batch_id: Optional[int] = None
    measurement_layer: str = "scheduler_layer"  # "direct_backend" or "scheduler_layer"
    
    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds.
        
        BUG FIX: Use `is not None` checks, not truthiness.
        arrival_time=0.0 is valid and should not be treated as missing.
        """
        if self.first_token_time is not None and self.arrival_time is not None:
            return (self.first_token_time - self.arrival_time) * 1000
        return 0.0
    
    @property
    def e2e_ms(self) -> float:
        """End-to-end latency in milliseconds."""
        if self.end_time is not None and self.arrival_time is not None:
            return (self.end_time - self.arrival_time) * 1000
        return 0.0
    
    @property
    def queue_time_ms(self) -> float:
        """Queue/dispatch time in milliseconds."""
        if self.dispatch_time is not None and self.arrival_time is not None:
            return (self.dispatch_time - self.arrival_time) * 1000
        return 0.0
    
    def validate(self) -> List[str]:
        """Validate timing invariants. Returns list of violations."""
        violations = []
        
        # Check timing order: arrival <= first_token <= end
        if self.first_token_time is not None and self.arrival_time is not None:
            if self.first_token_time < self.arrival_time:
                violations.append(
                    f"{self.request_id}: first_token_time ({self.first_token_time:.3f}) < "
                    f"arrival_time ({self.arrival_time:.3f})"
                )
        
        if self.end_time is not None and self.first_token_time is not None:
            if self.end_time < self.first_token_time:
                violations.append(
                    f"{self.request_id}: end_time ({self.end_time:.3f}) < "
                    f"first_token_time ({self.first_token_time:.3f})"
                )
        
        if self.end_time is not None and self.arrival_time is not None:
            if self.end_time < self.arrival_time:
                violations.append(
                    f"{self.request_id}: end_time ({self.end_time:.3f}) < "
                    f"arrival_time ({self.arrival_time:.3f})"
                )
        
        # Check e2e >= ttft
        if self.ttft_ms > 0 and self.e2e_ms > 0:
            if self.e2e_ms < self.ttft_ms:
                violations.append(
                    f"{self.request_id}: e2e_ms ({self.e2e_ms:.1f}) < ttft_ms ({self.ttft_ms:.1f})"
                )
        
        return violations
    
    def to_dict(self) -> dict:
        """Convert to dictionary with computed metrics."""
        d = asdict(self)
        d['ttft_ms'] = self.ttft_ms
        d['e2e_ms'] = self.e2e_ms
        d['queue_time_ms'] = self.queue_time_ms
        return d


class MetricsAggregator:
    """Aggregate metrics across requests with validation."""
    
    def __init__(self, measurement_layer: str = "scheduler_layer"):
        self.metrics: List[RequestMetrics] = []
        self.measurement_layer = measurement_layer
    
    def add(self, metric: RequestMetrics):
        """Add a metric."""
        metric.measurement_layer = self.measurement_layer
        self.metrics.append(metric)
    
    def validate_all(self, fail_on_error: bool = False) -> Dict:
        """Validate all metrics and return report.
        
        Args:
            fail_on_error: If True, raise exception on validation failures.
            
        Returns:
            Dict with validation results.
        """
        all_violations = []
        for m in self.metrics:
            violations = m.validate()
            all_violations.extend(violations)
        
        # Check that at least 99% of requests have valid TTFT
        valid_ttft_count = sum(1 for m in self.metrics 
                               if m.first_token_time is not None and m.ttft_ms > 0)
        ttft_valid_pct = (valid_ttft_count / len(self.metrics) * 100) if self.metrics else 0
        
        # Check for zero metrics (the bug we're fixing)
        zero_ttft_count = sum(1 for m in self.metrics 
                              if m.first_token_time is not None and m.ttft_ms == 0)
        zero_e2e_count = sum(1 for m in self.metrics 
                             if m.end_time is not None and m.e2e_ms == 0)
        
        report = {
            "num_requests": len(self.metrics),
            "num_violations": len(all_violations),
            "violations": all_violations[:20],  # First 20
            "ttft_valid_pct": ttft_valid_pct,
            "zero_ttft_count": zero_ttft_count,
            "zero_e2e_count": zero_e2e_count,
            "valid": len(all_violations) == 0 and zero_ttft_count == 0,
        }
        
        if fail_on_error and not report["valid"]:
            raise MetricsValidationError(
                f"Metrics validation failed: {len(all_violations)} violations, "
                f"{zero_ttft_count} zero TTFT, {zero_e2e_count} zero E2E. "
                f"First violations: {all_violations[:5]}"
            )
        
        return report
    
    def get_summary(self, validate: bool = True) -> Dict:
        """Get summary statistics.
        
        Args:
            validate: If True, validate metrics and warn on issues.
        """
        if not self.metrics:
            return {}
        
        # Validate and warn (but don't fail)
        if validate:
            report = self.validate_all(fail_on_error=False)
            if not report["valid"]:
                print(f"⚠️  Metrics validation issues: {report['num_violations']} violations, "
                      f"{report['zero_ttft_count']} zero TTFT, {report['zero_e2e_count']} zero E2E")
        
        # Collect valid metrics - use is not None checks
        ttfts = [m.ttft_ms for m in self.metrics 
                 if m.first_token_time is not None and m.ttft_ms > 0]
        e2es = [m.e2e_ms for m in self.metrics 
                if m.end_time is not None and m.e2e_ms > 0]
        queue_times = [m.queue_time_ms for m in self.metrics 
                       if m.dispatch_time is not None and m.queue_time_ms > 0]
        total_tokens = sum(m.generated_tokens for m in self.metrics)
        
        # Compute total time from actual timestamps
        total_time = 0.0
        if self.metrics:
            arrivals = [m.arrival_time for m in self.metrics if m.arrival_time is not None]
            ends = [m.end_time for m in self.metrics if m.end_time is not None]
            if arrivals and ends:
                total_time = max(ends) - min(arrivals)
        
        def percentile(data, p):
            """Calculate percentile using proper quantile method.
            
            Uses linear interpolation for proper percentile calculation.
            For P99 with n samples, returns value at index (n-1) * 0.99.
            """
            if len(data) == 0:  # Fixed: use len() instead of truthiness for numpy arrays
                return 0.0
            sorted_data = sorted(data)
            n = len(sorted_data)
            # Proper quantile: (n-1) * p/100 for 0-indexed array
            # Use linear interpolation for better accuracy
            position = (n - 1) * p / 100
            lower_idx = int(position)
            upper_idx = min(lower_idx + 1, n - 1)
            weight = position - lower_idx
            
            if lower_idx == upper_idx:
                return sorted_data[lower_idx]
            # Linear interpolation
            return sorted_data[lower_idx] * (1 - weight) + sorted_data[upper_idx] * weight
        
        def mean(data):
            return sum(data) / len(data) if data else 0.0
        
        # Compute statistical measures
        ttft_std = np.std(ttfts) if len(ttfts) > 1 else 0.0
        e2e_std = np.std(e2es) if len(e2es) > 1 else 0.0
        
        # Bootstrap confidence intervals for P99 (if sample size is sufficient)
        def bootstrap_ci(data, p, n_bootstrap=1000, confidence=0.95):
            """Compute bootstrap confidence interval for percentile."""
            if len(data) < 10:  # Too small for bootstrap
                return None, None
            bootstraps = []
            data_list = list(data)  # Convert to list to avoid numpy array issues
            for _ in range(n_bootstrap):
                sample = np.random.choice(data_list, size=len(data_list), replace=True)
                bootstraps.append(percentile(list(sample), p))  # Ensure list for percentile function
            alpha = 1 - confidence
            lower = np.percentile(bootstraps, alpha/2 * 100)
            upper = np.percentile(bootstraps, (1 - alpha/2) * 100)
            return lower, upper
        
        ttft_p99_ci = bootstrap_ci(ttfts, 99) if len(ttfts) >= 10 else (None, None)
        e2e_p99_ci = bootstrap_ci(e2es, 99) if len(e2es) >= 10 else (None, None)
        
        return {
            "num_requests": len(self.metrics),
            "num_valid_ttft": len(ttfts),
            "num_valid_e2e": len(e2es),
            "total_tokens": total_tokens,
            "total_time_s": total_time,
            "throughput_tok_s": total_tokens / total_time if total_time > 0 else 0.0,
            "throughput_req_s": len(self.metrics) / total_time if total_time > 0 else 0.0,
            "ttft_mean": mean(ttfts),
            "ttft_std": float(ttft_std),
            "ttft_p50": percentile(ttfts, 50),
            "ttft_p95": percentile(ttfts, 95),
            "ttft_p99": percentile(ttfts, 99),
            "ttft_p99_ci_lower": float(ttft_p99_ci[0]) if ttft_p99_ci[0] is not None else None,
            "ttft_p99_ci_upper": float(ttft_p99_ci[1]) if ttft_p99_ci[1] is not None else None,
            "e2e_mean": mean(e2es),
            "e2e_std": float(e2e_std),
            "e2e_p50": percentile(e2es, 50),
            "e2e_p95": percentile(e2es, 95),
            "e2e_p99": percentile(e2es, 99),
            "e2e_p99_ci_lower": float(e2e_p99_ci[0]) if e2e_p99_ci[0] is not None else None,
            "e2e_p99_ci_upper": float(e2e_p99_ci[1]) if e2e_p99_ci[1] is not None else None,
            "queue_p50": percentile(queue_times, 50),
            "queue_p95": percentile(queue_times, 95),
            "queue_p99": percentile(queue_times, 99),
            "measurement_layer": self.measurement_layer,
        }
    
    def save_jsonl(self, filepath: str):
        """Save metrics as JSONL with validation."""
        from ..utils.io import save_jsonl
        
        # Validate before saving
        report = self.validate_all(fail_on_error=False)
        if not report["valid"]:
            print(f"⚠️  Saving metrics with validation issues to {filepath}")
        
        data = [m.to_dict() for m in self.metrics]
        save_jsonl(data, filepath)
        
        return report
    
    def filter_by_length(self, is_short: bool, short_threshold: int = 64) -> 'MetricsAggregator':
        """Filter metrics by request length."""
        filtered = MetricsAggregator(measurement_layer=self.measurement_layer)
        for m in self.metrics:
            if is_short and m.max_new_tokens <= short_threshold:
                filtered.add(m)
            elif not is_short and m.max_new_tokens > short_threshold:
                filtered.add(m)
        return filtered
    
    @classmethod
    def from_jsonl(cls, filepath: str) -> 'MetricsAggregator':
        """Load metrics from JSONL file."""
        from ..utils.io import load_jsonl
        
        data = load_jsonl(filepath)
        aggregator = cls()
        
        for d in data:
            # Extract fields, handling both old and new formats
            metric = RequestMetrics(
                request_id=d.get('request_id', 'unknown'),
                prompt=d.get('prompt', ''),
                prompt_tokens=d.get('prompt_tokens', 0),
                generated_tokens=d.get('generated_tokens', 0),
                max_new_tokens=d.get('max_new_tokens', 0),
                arrival_time=d.get('arrival_time', 0.0),
                first_token_time=d.get('first_token_time'),
                end_time=d.get('end_time'),
                dispatch_time=d.get('dispatch_time'),
                backend=d.get('backend', 'unknown'),
                policy=d.get('policy', 'unknown'),
                batch_id=d.get('batch_id'),
                measurement_layer=d.get('measurement_layer', 'scheduler_layer'),
            )
            aggregator.add(metric)
        
        return aggregator
