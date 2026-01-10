"""Validation utilities for experiment logs.

This module provides functions to validate JSONL log files and
ensure metrics are correctly computed.

Usage:
    from src.utils.validate_logs import validate_jsonl_logs, recompute_metrics
    
    # Validate existing logs
    report = validate_jsonl_logs("/results/runs/intervention_fifo.jsonl")
    
    # Recompute metrics from raw logs
    summary = recompute_metrics("/results/runs/intervention_fifo.jsonl")
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidationReport:
    """Report from log validation."""
    filepath: str
    num_records: int
    valid: bool
    issues: List[str]
    zero_ttft_count: int
    zero_e2e_count: int
    negative_ttft_count: int
    negative_e2e_count: int
    timing_order_violations: int
    sample_records: List[Dict]  # First few records for debugging


def validate_jsonl_logs(filepath: str) -> ValidationReport:
    """Validate a JSONL log file for common issues.
    
    Checks:
    1. All records have required fields
    2. No zero TTFT/E2E values (unless expected)
    3. No negative timing values
    4. Timing order: arrival <= first_token <= end
    
    Returns:
        ValidationReport with detailed findings.
    """
    issues = []
    records = []
    
    path = Path(filepath)
    if not path.exists():
        return ValidationReport(
            filepath=filepath,
            num_records=0,
            valid=False,
            issues=[f"File not found: {filepath}"],
            zero_ttft_count=0,
            zero_e2e_count=0,
            negative_ttft_count=0,
            negative_e2e_count=0,
            timing_order_violations=0,
            sample_records=[],
        )
    
    # Load records
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                issues.append(f"Line {line_num}: JSON parse error: {e}")
    
    if not records:
        return ValidationReport(
            filepath=filepath,
            num_records=0,
            valid=False,
            issues=["File is empty"],
            zero_ttft_count=0,
            zero_e2e_count=0,
            negative_ttft_count=0,
            negative_e2e_count=0,
            timing_order_violations=0,
            sample_records=[],
        )
    
    # Required fields
    required_fields = ['request_id', 'arrival_time', 'first_token_time', 'end_time']
    
    zero_ttft = 0
    zero_e2e = 0
    negative_ttft = 0
    negative_e2e = 0
    timing_violations = 0
    
    for i, record in enumerate(records):
        rid = record.get('request_id', f'record_{i}')
        
        # Check required fields
        for field in required_fields:
            if field not in record:
                issues.append(f"{rid}: Missing field '{field}'")
        
        # Extract timing values
        arrival = record.get('arrival_time')
        first_token = record.get('first_token_time')
        end = record.get('end_time')
        
        # Check for pre-computed vs raw values
        ttft_ms = record.get('ttft_ms')
        e2e_ms = record.get('e2e_ms')
        
        # If pre-computed values exist, check them
        if ttft_ms is not None:
            if ttft_ms == 0 and first_token is not None:
                zero_ttft += 1
                issues.append(f"{rid}: Zero TTFT with non-null first_token_time (BUG)")
            elif ttft_ms < 0:
                negative_ttft += 1
                issues.append(f"{rid}: Negative TTFT: {ttft_ms}")
        
        if e2e_ms is not None:
            if e2e_ms == 0 and end is not None:
                zero_e2e += 1
                issues.append(f"{rid}: Zero E2E with non-null end_time (BUG)")
            elif e2e_ms < 0:
                negative_e2e += 1
                issues.append(f"{rid}: Negative E2E: {e2e_ms}")
        
        # Check timing order
        if arrival is not None and first_token is not None:
            if first_token < arrival:
                timing_violations += 1
                issues.append(f"{rid}: first_token ({first_token:.3f}) < arrival ({arrival:.3f})")
        
        if first_token is not None and end is not None:
            if end < first_token:
                timing_violations += 1
                issues.append(f"{rid}: end ({end:.3f}) < first_token ({first_token:.3f})")
    
    # Limit issues list for report
    is_valid = (
        len(issues) == 0 and 
        zero_ttft == 0 and 
        zero_e2e == 0 and
        timing_violations == 0
    )
    
    return ValidationReport(
        filepath=filepath,
        num_records=len(records),
        valid=is_valid,
        issues=issues[:50],  # First 50 issues
        zero_ttft_count=zero_ttft,
        zero_e2e_count=zero_e2e,
        negative_ttft_count=negative_ttft,
        negative_e2e_count=negative_e2e,
        timing_order_violations=timing_violations,
        sample_records=records[:3],  # First 3 records for debugging
    )


def recompute_metrics(filepath: str) -> Dict:
    """Recompute summary metrics from raw JSONL logs.
    
    This bypasses any cached/stored metrics and recomputes from
    the raw timing values in the log file.
    
    Returns:
        Dict with recomputed metrics.
    """
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    records = []
    with open(path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if not records:
        return {"error": "No valid records"}
    
    # Recompute TTFT and E2E from raw timing values
    ttfts = []
    e2es = []
    total_tokens = 0
    
    arrivals = []
    ends = []
    
    for r in records:
        arrival = r.get('arrival_time')
        first_token = r.get('first_token_time')
        end = r.get('end_time')
        
        # Compute TTFT (in ms)
        if arrival is not None and first_token is not None:
            ttft = (first_token - arrival) * 1000
            if ttft > 0:  # Only include valid positive values
                ttfts.append(ttft)
        
        # Compute E2E (in ms)
        if arrival is not None and end is not None:
            e2e = (end - arrival) * 1000
            if e2e > 0:
                e2es.append(e2e)
        
        # Track for total time
        if arrival is not None:
            arrivals.append(arrival)
        if end is not None:
            ends.append(end)
        
        # Sum tokens
        total_tokens += r.get('generated_tokens', 0)
    
    # Compute total wall-clock time
    total_time = 0.0
    if arrivals and ends:
        total_time = max(ends) - min(arrivals)
    
    def percentile(data, p):
        """Calculate percentile using proper quantile method with linear interpolation."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        # Proper quantile: (n-1) * p/100 for 0-indexed array
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
    
    return {
        "num_records": len(records),
        "num_valid_ttft": len(ttfts),
        "num_valid_e2e": len(e2es),
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "throughput_tok_s": total_tokens / total_time if total_time > 0 else 0.0,
        "ttft_mean": mean(ttfts),
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p95": percentile(ttfts, 95),
        "ttft_p99": percentile(ttfts, 99),
        "e2e_mean": mean(e2es),
        "e2e_p50": percentile(e2es, 50),
        "e2e_p95": percentile(e2es, 95),
        "e2e_p99": percentile(e2es, 99),
    }


def filter_by_length(records: List[Dict], is_short: bool, threshold: int = 64) -> List[Dict]:
    """Filter records by max_new_tokens."""
    filtered = []
    for r in records:
        max_tokens = r.get('max_new_tokens', 0)
        if is_short and max_tokens <= threshold:
            filtered.append(r)
        elif not is_short and max_tokens > threshold:
            filtered.append(r)
    return filtered


def analyze_intervention_logs(
    fifo_path: str,
    intervention_path: str,
    short_threshold: int = 64
) -> Dict:
    """Analyze intervention experiment logs and compute improvement.
    
    Args:
        fifo_path: Path to FIFO baseline JSONL logs
        intervention_path: Path to intervention JSONL logs
        short_threshold: Max tokens threshold for "short" requests
        
    Returns:
        Dict with comparison metrics and improvement percentage.
    """
    # Load and validate both log files
    fifo_report = validate_jsonl_logs(fifo_path)
    intervention_report = validate_jsonl_logs(intervention_path)
    
    results = {
        "fifo_validation": {
            "valid": fifo_report.valid,
            "num_records": fifo_report.num_records,
            "issues_count": len(fifo_report.issues),
        },
        "intervention_validation": {
            "valid": intervention_report.valid,
            "num_records": intervention_report.num_records,
            "issues_count": len(intervention_report.issues),
        },
    }
    
    # Load raw records
    fifo_records = []
    intervention_records = []
    
    if Path(fifo_path).exists():
        with open(fifo_path, 'r') as f:
            fifo_records = [json.loads(line) for line in f if line.strip()]
    
    if Path(intervention_path).exists():
        with open(intervention_path, 'r') as f:
            intervention_records = [json.loads(line) for line in f if line.strip()]
    
    # Filter for short requests
    fifo_short = filter_by_length(fifo_records, is_short=True, threshold=short_threshold)
    intervention_short = filter_by_length(intervention_records, is_short=True, threshold=short_threshold)
    
    # Compute metrics for short requests
    def compute_e2e_p99(records: List[Dict]) -> float:
        """Compute P99 E2E latency with proper percentile calculation."""
        e2es = []
        for r in records:
            arrival = r.get('arrival_time')
            end = r.get('end_time')
            if arrival is not None and end is not None:
                e2e = (end - arrival) * 1000
                if e2e > 0:
                    e2es.append(e2e)
        if not e2es:
            return 0.0
        sorted_e2es = sorted(e2es)
        n = len(sorted_e2es)
        # Proper quantile: (n-1) * 0.99 for P99
        position = (n - 1) * 0.99
        lower_idx = int(position)
        upper_idx = min(lower_idx + 1, n - 1)
        weight = position - lower_idx
        
        if lower_idx == upper_idx:
            return sorted_e2es[lower_idx]
        # Linear interpolation
        return sorted_e2es[lower_idx] * (1 - weight) + sorted_e2es[upper_idx] * weight
    
    fifo_short_e2e_p99 = compute_e2e_p99(fifo_short)
    intervention_short_e2e_p99 = compute_e2e_p99(intervention_short)
    
    # Compute improvement
    if fifo_short_e2e_p99 > 0:
        improvement_pct = (fifo_short_e2e_p99 - intervention_short_e2e_p99) / fifo_short_e2e_p99 * 100
    else:
        improvement_pct = 0.0
    
    results["fifo_all"] = recompute_metrics(fifo_path)
    results["intervention_all"] = recompute_metrics(intervention_path)
    results["fifo_short"] = {
        "num_requests": len(fifo_short),
        "e2e_p99": fifo_short_e2e_p99,
    }
    results["intervention_short"] = {
        "num_requests": len(intervention_short),
        "e2e_p99": intervention_short_e2e_p99,
    }
    results["improvement_pct"] = improvement_pct
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Validating: {filepath}")
        report = validate_jsonl_logs(filepath)
        print(f"Valid: {report.valid}")
        print(f"Records: {report.num_records}")
        print(f"Zero TTFT: {report.zero_ttft_count}")
        print(f"Zero E2E: {report.zero_e2e_count}")
        print(f"Issues: {len(report.issues)}")
        if report.issues:
            print("First 5 issues:")
            for issue in report.issues[:5]:
                print(f"  - {issue}")
        
        print("\nRecomputed metrics:")
        metrics = recompute_metrics(filepath)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
