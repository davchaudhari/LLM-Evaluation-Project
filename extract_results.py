#!/usr/bin/env python3
"""Extract and analyze results from Modal volume."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiments.metrics import MetricsAggregator
from utils.io import load_json

def analyze_intervention():
    """Analyze intervention experiment from JSONL files."""
    print("=" * 80)
    print("INTERVENTION EXPERIMENT RESULTS")
    print("=" * 80)
    
    # Try to load from Modal volume
    baseline_path = "/results/runs/intervention_fifo.jsonl"
    intervention_path = "/results/runs/intervention_length_aware.jsonl"
    
    if not Path(baseline_path).exists():
        print(f"❌ Baseline log not found: {baseline_path}")
        return None
    
    if not Path(intervention_path).exists():
        print(f"❌ Intervention log not found: {intervention_path}")
        return None
    
    # Load metrics
    baseline_metrics = MetricsAggregator.from_jsonl(baseline_path)
    intervention_metrics = MetricsAggregator.from_jsonl(intervention_path)
    
    baseline_summary = baseline_metrics.get_summary(validate=True)
    intervention_summary = intervention_metrics.get_summary(validate=True)
    
    # Filter short requests (max_new_tokens <= 64)
    baseline_short = MetricsAggregator([m for m in baseline_metrics.metrics if m.max_new_tokens <= 64])
    intervention_short = MetricsAggregator([m for m in intervention_metrics.metrics if m.max_new_tokens <= 64])
    
    baseline_short_summary = baseline_short.get_summary(validate=True)
    intervention_short_summary = intervention_short.get_summary(validate=True)
    
    print(f"\n📊 BASELINE (FIFO Policy)")
    print(f"   Total Requests: {baseline_summary['num_requests']}")
    print(f"   Short Requests: {baseline_short_summary['num_requests']}")
    print(f"   Throughput: {baseline_summary['throughput_tok_s']:.1f} tok/s")
    print(f"   All Requests - TTFT P99: {baseline_summary['ttft_p99']:.1f} ms, E2E P99: {baseline_summary['e2e_p99']:.1f} ms")
    print(f"   Short Requests - TTFT P99: {baseline_short_summary['ttft_p99']:.1f} ms, E2E P99: {baseline_short_summary['e2e_p99']:.1f} ms")
    
    print(f"\n📊 INTERVENTION (Length-Aware Microbatch)")
    print(f"   Total Requests: {intervention_summary['num_requests']}")
    print(f"   Short Requests: {intervention_short_summary['num_requests']}")
    print(f"   Throughput: {intervention_summary['throughput_tok_s']:.1f} tok/s")
    print(f"   All Requests - TTFT P99: {intervention_summary['ttft_p99']:.1f} ms, E2E P99: {intervention_summary['e2e_p99']:.1f} ms")
    print(f"   Short Requests - TTFT P99: {intervention_short_summary['ttft_p99']:.1f} ms, E2E P99: {intervention_short_summary['e2e_p99']:.1f} ms")
    
    # Calculate improvement
    baseline_e2e = baseline_short_summary['e2e_p99']
    intervention_e2e = intervention_short_summary['e2e_p99']
    
    if baseline_e2e > 0:
        improvement = (baseline_e2e - intervention_e2e) / baseline_e2e * 100
        print(f"\n🎯 IMPROVEMENT: {improvement:.1f}%")
        print(f"   Baseline Short E2E P99: {baseline_e2e:.1f} ms")
        print(f"   Intervention Short E2E P99: {intervention_e2e:.1f} ms")
        print(f"   Absolute Improvement: {baseline_e2e - intervention_e2e:.1f} ms")
        
        if improvement >= 50:
            print(f"\n✅ SUCCESS: Met target of ≥50% improvement!")
        elif improvement > 0:
            print(f"\n⚠️  Improvement below 50% target")
        else:
            print(f"\n❌ No improvement or regression")
    else:
        print(f"\n⚠️  Cannot calculate improvement: baseline E2E P99 is 0")
        improvement = None
    
    # Statistical analysis
    print(f"\n📈 STATISTICAL ANALYSIS")
    try:
        from scipy import stats
        import numpy as np
        
        baseline_e2es = [m.e2e_ms for m in baseline_short.metrics if m.e2e_ms > 0]
        intervention_e2es = [m.e2e_ms for m in intervention_short.metrics if m.e2e_ms > 0]
        
        if len(baseline_e2es) > 0 and len(intervention_e2es) > 0:
            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(
                baseline_e2es,
                intervention_e2es,
                alternative='greater'
            )
            
            print(f"   Mann-Whitney U Test:")
            print(f"     Statistic: {statistic:.2f}")
            print(f"     p-value: {p_value:.6f}")
            print(f"     Significant (p < 0.05): {'✅ YES' if p_value < 0.05 else '❌ NO'}")
            
            # Effect size
            baseline_mean = np.mean(baseline_e2es)
            intervention_mean = np.mean(intervention_e2es)
            pooled_std = np.sqrt((np.var(baseline_e2es) + np.var(intervention_e2es)) / 2)
            cohens_d = (baseline_mean - intervention_mean) / pooled_std if pooled_std > 0 else 0
            
            print(f"   Effect Size (Cohen's d): {cohens_d:.3f}")
            if abs(cohens_d) < 0.2:
                print("     Interpretation: Negligible")
            elif abs(cohens_d) < 0.5:
                print("     Interpretation: Small")
            elif abs(cohens_d) < 0.8:
                print("     Interpretation: Medium")
            else:
                print("     Interpretation: Large")
            
            print(f"   Baseline Mean E2E: {baseline_mean:.1f} ms")
            print(f"   Intervention Mean E2E: {intervention_mean:.1f} ms")
        else:
            print("   ⚠️  Insufficient data for statistical analysis")
    except ImportError:
        print("   ⚠️  scipy not available")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    return {
        "baseline": baseline_summary,
        "intervention": intervention_summary,
        "baseline_short": baseline_short_summary,
        "intervention_short": intervention_short_summary,
        "improvement_pct": improvement,
    }


# This function should be added to modal_app.py
# For now, we'll create a standalone Modal function
def extract_and_report_modal():
    """Extract results and generate report."""
    import sys
    sys.path.insert(0, "/root")
    
    from pathlib import Path
    
    results = analyze_intervention()
    
    # Also try to load other results
    print("\n" + "=" * 80)
    print("OTHER RESULTS")
    print("=" * 80)
    
    try:
        hf_results = load_json("/results/hf_suite_summary.json")
        print(f"\n✅ HF Suite: {len(hf_results.get('dispatch_policies', []))} policies tested")
    except:
        print("\n⚠️  HF Suite results not available")
    
    try:
        vllm_results = load_json("/results/vllm_suite_summary.json")
        print(f"✅ vLLM Suite: {len(vllm_results.get('dispatch_policies', []))} policies tested")
    except:
        print("⚠️  vLLM Suite results not available")
    
    return results
