#!/usr/bin/env python3
"""
Analyze experimental results from Modal and generate comprehensive report.

This script:
1. Downloads/accesses results from Modal volume
2. Analyzes intervention experiment with proper statistics
3. Generates a comprehensive results report
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiments.metrics import MetricsAggregator
from utils.validate_logs import analyze_intervention_logs


def load_results_from_modal() -> Dict[str, Any]:
    """Load results from Modal volume (run this inside Modal function)."""
    results = {}
    
    # Try to load intervention results
    intervention_path = "/results/intervention_results.json"
    if Path(intervention_path).exists():
        with open(intervention_path, 'r') as f:
            results['intervention'] = json.load(f)
    
    # Try to load raw JSONL logs
    baseline_log = "/results/runs/intervention_fifo.jsonl"
    intervention_log = "/results/runs/intervention_length_aware.jsonl"
    
    if Path(baseline_log).exists() and Path(intervention_log).exists():
        results['baseline_log'] = baseline_log
        results['intervention_log'] = intervention_log
        
        # Load and analyze
        baseline_metrics = MetricsAggregator.from_jsonl(baseline_log)
        intervention_metrics = MetricsAggregator.from_jsonl(intervention_log)
        
        results['baseline_metrics'] = baseline_metrics
        results['intervention_metrics'] = intervention_metrics
    
    return results


def generate_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive results report."""
    report = []
    report.append("# Experimental Results Report")
    report.append("**Generated**: 2026-01-06")
    report.append("**Status**: Post-Fix Validation with Statistical Analysis")
    report.append("")
    
    if 'intervention' in results:
        interv = results['intervention']
        report.append("## Intervention Experiment Results")
        report.append("")
        
        if 'baseline' in interv and 'intervention' in interv:
            base = interv['baseline']
            inter = interv['intervention']
            
            report.append("### Workload Configuration")
            report.append(f"- **Total Requests**: {base.get('num_requests', 'N/A')}")
            report.append(f"- **Short Requests**: {base.get('short_num_requests', 'N/A')} (32 tokens)")
            report.append(f"- **Long Requests**: {base.get('num_requests', 0) - base.get('short_num_requests', 0)} (256 tokens)")
            report.append(f"- **Model**: Qwen/Qwen2.5-3B-Instruct")
            report.append(f"- **Backend**: vLLM")
            report.append("")
            
            report.append("### Baseline (FIFO Policy)")
            report.append(f"- **Throughput**: {base.get('throughput_tok_s', 0):.1f} tok/s")
            report.append(f"- **All Requests E2E P99**: {base.get('e2e_p99', 0):.1f} ms")
            report.append(f"- **Short Requests E2E P99**: {base.get('short_e2e_p99', 0):.1f} ms")
            report.append(f"- **Short Requests TTFT P99**: {base.get('short_ttft_p99', 0):.1f} ms")
            report.append("")
            
            report.append("### Intervention (Length-Aware Microbatch)")
            report.append(f"- **Throughput**: {inter.get('throughput_tok_s', 0):.1f} tok/s")
            report.append(f"- **All Requests E2E P99**: {inter.get('e2e_p99', 0):.1f} ms")
            report.append(f"- **Short Requests E2E P99**: {inter.get('short_e2e_p99', 0):.1f} ms")
            report.append(f"- **Short Requests TTFT P99**: {inter.get('short_ttft_p99', 0):.1f} ms")
            report.append("")
            
            improvement = interv.get('improvement_pct')
            if improvement and improvement != "INVALID":
                report.append(f"### Improvement: {improvement:.1f}%")
                report.append("")
                
                baseline_e2e = base.get('short_e2e_p99', 0)
                intervention_e2e = inter.get('short_e2e_p99', 0)
                report.append(f"- **Baseline Short E2E P99**: {baseline_e2e:.1f} ms")
                report.append(f"- **Intervention Short E2E P99**: {intervention_e2e:.1f} ms")
                report.append(f"- **Absolute Improvement**: {baseline_e2e - intervention_e2e:.1f} ms")
                report.append("")
            
            # Statistical analysis
            if 'statistical_analysis' in interv and interv['statistical_analysis']:
                stats = interv['statistical_analysis']
                report.append("### Statistical Analysis")
                report.append("")
                report.append(f"- **Mann-Whitney U Test**:")
                report.append(f"  - Statistic: {stats.get('mann_whitney_u', 'N/A')}")
                report.append(f"  - p-value: {stats.get('p_value', 'N/A'):.6f}")
                report.append(f"  - Significant (p < 0.05): {'✅ YES' if stats.get('significant') else '❌ NO'}")
                report.append("")
                report.append(f"- **Effect Size (Cohen's d)**: {stats.get('cohens_d', 'N/A'):.3f}")
                cohens_d = stats.get('cohens_d', 0)
                if abs(cohens_d) < 0.2:
                    report.append("  - Interpretation: Negligible")
                elif abs(cohens_d) < 0.5:
                    report.append("  - Interpretation: Small")
                elif abs(cohens_d) < 0.8:
                    report.append("  - Interpretation: Medium")
                else:
                    report.append("  - Interpretation: Large")
                report.append("")
                report.append(f"- **Baseline Mean E2E**: {stats.get('baseline_mean', 'N/A'):.1f} ms")
                report.append(f"- **Intervention Mean E2E**: {stats.get('intervention_mean', 'N/A'):.1f} ms")
                report.append("")
            
            # Confidence intervals
            if 'e2e_p99_ci_lower' in base and base.get('e2e_p99_ci_lower') is not None:
                report.append("### Confidence Intervals (Bootstrap 95% CI)")
                report.append("")
                report.append("**Baseline Short E2E P99**:")
                report.append(f"- Lower: {base.get('e2e_p99_ci_lower', 'N/A'):.1f} ms")
                report.append(f"- Upper: {base.get('e2e_p99_ci_upper', 'N/A'):.1f} ms")
                report.append("")
                report.append("**Intervention Short E2E P99**:")
                report.append(f"- Lower: {inter.get('e2e_p99_ci_lower', 'N/A'):.1f} ms")
                report.append(f"- Upper: {inter.get('e2e_p99_ci_upper', 'N/A'):.1f} ms")
                report.append("")
            
            # Validation
            if 'validation' in interv:
                report.append("### Validation Status")
                report.append("")
                base_val = interv['validation'].get('baseline', {})
                inter_val = interv['validation'].get('intervention', {})
                report.append(f"- **Baseline**: {base_val.get('num_requests', 'N/A')} requests, valid={base_val.get('valid', 'N/A')}")
                report.append(f"- **Intervention**: {inter_val.get('num_requests', 'N/A')} requests, valid={inter_val.get('valid', 'N/A')}")
                report.append(f"- **Comparison Valid**: {interv.get('comparison_valid', 'N/A')}")
                report.append("")
    
    # Additional metrics if available
    if 'baseline_metrics' in results and 'intervention_metrics' in results:
        baseline = results['baseline_metrics']
        intervention = results['intervention_metrics']
        
        baseline_summary = baseline.get_summary(validate=True)
        intervention_summary = intervention.get_summary(validate=True)
        
        report.append("## Detailed Metrics")
        report.append("")
        report.append("### Baseline Summary Statistics")
        report.append(f"- **TTFT Mean**: {baseline_summary.get('ttft_mean', 0):.1f} ms")
        report.append(f"- **TTFT Std Dev**: {baseline_summary.get('ttft_std', 0):.1f} ms")
        report.append(f"- **E2E Mean**: {baseline_summary.get('e2e_mean', 0):.1f} ms")
        report.append(f"- **E2E Std Dev**: {baseline_summary.get('e2e_std', 0):.1f} ms")
        report.append("")
        report.append("### Intervention Summary Statistics")
        report.append(f"- **TTFT Mean**: {intervention_summary.get('ttft_mean', 0):.1f} ms")
        report.append(f"- **TTFT Std Dev**: {intervention_summary.get('ttft_std', 0):.1f} ms")
        report.append(f"- **E2E Mean**: {intervention_summary.get('e2e_mean', 0):.1f} ms")
        report.append(f"- **E2E Std Dev**: {intervention_summary.get('e2e_std', 0):.1f} ms")
        report.append("")
    
    report.append("## Fixes Applied")
    report.append("")
    report.append("✅ **Percentile Calculation**: Fixed to use proper quantile method with linear interpolation")
    report.append("✅ **Sample Size**: Increased from 32 to 1000 requests (500 short + 500 long)")
    report.append("✅ **Statistical Testing**: Added Mann-Whitney U test, effect size, confidence intervals")
    report.append("✅ **TTFT Measurement**: Fixed to use true streaming TTFT (AsyncLLMEngine)")
    report.append("✅ **Bootstrap CIs**: Added 95% confidence intervals for P99 metrics")
    report.append("")
    
    report.append("## Conclusion")
    report.append("")
    if 'intervention' in results:
        improvement = results['intervention'].get('improvement_pct')
        if improvement and improvement != "INVALID":
            stats = results['intervention'].get('statistical_analysis')
            if stats and stats.get('significant'):
                report.append(f"✅ **Statistically Significant Improvement**: {improvement:.1f}% reduction in short-request E2E P99")
                report.append(f"   - p-value: {stats.get('p_value', 0):.6f} < 0.05")
                report.append(f"   - Effect size: {stats.get('cohens_d', 0):.3f}")
            else:
                report.append(f"⚠️  **Improvement Observed**: {improvement:.1f}%, but statistical significance not confirmed")
        else:
            report.append("❌ **Comparison Invalid**: Results cannot be compared due to workload mismatch")
    
    return "\n".join(report)


# Note: This function should be added to modal_app.py
# For now, this is a standalone script
def generate_results_report_modal():
    """Generate comprehensive results report from Modal."""
    import sys
    sys.path.insert(0, "/root")
    
    from pathlib import Path
    
    # Load results
    results = load_results_from_modal()
    
    # Generate report
    report = generate_report(results)
    
    # Save report
    report_path = "/results/EXPERIMENTAL_RESULTS_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print("=" * 80)
    print("EXPERIMENTAL RESULTS REPORT")
    print("=" * 80)
    print(report)
    print("=" * 80)
    print(f"\n✓ Report saved to {report_path}")
    
    return {"report_path": report_path, "results": results}


if __name__ == "__main__":
    # For local testing
    print("This script should be run via Modal:")
    print("  modal run modal_app.py::generate_results_report")
    print("\nOr import and use the functions directly in Modal functions.")
