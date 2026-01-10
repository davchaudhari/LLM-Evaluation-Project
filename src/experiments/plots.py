"""Plotting utilities."""

import matplotlib.pyplot as plt
from typing import List, Dict
from pathlib import Path


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile."""
    if not data:
        return 0.0
    idx = int(len(data) * p / 100)
    return sorted(data)[min(idx, len(data) - 1)]


def plot_ttft_stair_step(metrics_list: List[Dict], save_path: str):
    """Plot TTFT vs request position (stair-step)."""
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    request_ids = [i for i in range(len(metrics_list))]
    ttfts = [m.get("ttft_ms", 0) for m in metrics_list]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by batch (assuming batch_size=8)
    batch_size = 8
    colors = ['#2ecc71' if i < batch_size else '#e74c3c' for i in request_ids]
    ax.bar(request_ids, ttfts, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=batch_size - 0.5, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Request Arrival Order')
    ax.set_ylabel('Time to First Token (ms)')
    ax.set_title('TTFT vs Request Position (Stair-Step)')
    ax.set_xticks(request_ids)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_policy_comparison(results: List[Dict], save_path: str):
    """Plot dispatch policy comparison."""
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    policies = [r["policy"] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Throughput
    axes[0].bar(range(len(policies)), [r["throughput_tok_s"] for r in results], color='steelblue')
    axes[0].set_xticks(range(len(policies)))
    axes[0].set_xticklabels(policies, rotation=45, ha='right')
    axes[0].set_ylabel('Tokens/sec')
    axes[0].set_title('Throughput')
    
    # TTFT
    x = range(len(policies))
    width = 0.35
    axes[1].bar([i - width/2 for i in x], [r["ttft_p50"] for r in results], width, label='P50', color='green', alpha=0.7)
    axes[1].bar([i + width/2 for i in x], [r["ttft_p99"] for r in results], width, label='P99', color='red', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(policies, rotation=45, ha='right')
    axes[1].set_ylabel('TTFT (ms)')
    axes[1].set_title('Time to First Token')
    axes[1].legend()
    
    # E2E
    axes[2].bar([i - width/2 for i in x], [r["e2e_p50"] for r in results], width, label='P50', color='green', alpha=0.7)
    axes[2].bar([i + width/2 for i in x], [r["e2e_p99"] for r in results], width, label='P99', color='red', alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(policies, rotation=45, ha='right')
    axes[2].set_ylabel('E2E Latency (ms)')
    axes[2].set_title('End-to-End Latency')
    axes[2].legend()
    
    plt.suptitle('Dispatch Policy Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_hol_comparison(fifo_results: Dict, short_first_results: Dict, save_path: str):
    """Plot head-of-line blocking comparison."""
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = ['FIFO', 'Short-first']
    width = 0.35
    
    # E2E comparison
    short_e2e = [fifo_results["short_e2e_p99"], short_first_results["short_e2e_p99"]]
    long_e2e = [fifo_results["long_e2e_p99"], short_first_results["long_e2e_p99"]]
    
    axes[0].bar([0 - width/2, 1 - width/2], short_e2e, width, label='Short (32 tok)', color='#3498db')
    axes[0].bar([0 + width/2, 1 + width/2], long_e2e, width, label='Long (256 tok)', color='#e74c3c')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(x)
    axes[0].set_ylabel('E2E Latency P99 (ms)')
    axes[0].set_title('End-to-End Latency by Request Type')
    axes[0].legend()
    
    # TTFT comparison
    short_ttft = [fifo_results.get("short_ttft_p99", 0), short_first_results.get("short_ttft_p99", 0)]
    long_ttft = [fifo_results.get("long_ttft_p99", 0), short_first_results.get("long_ttft_p99", 0)]
    
    axes[1].bar([0 - width/2, 1 - width/2], short_ttft, width, label='Short (32 tok)', color='#3498db')
    axes[1].bar([0 + width/2, 1 + width/2], long_ttft, width, label='Long (256 tok)', color='#e74c3c')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(x)
    axes[1].set_ylabel('TTFT P99 (ms)')
    axes[1].set_title('Time to First Token by Request Type')
    axes[1].legend()
    
    plt.suptitle('Head-of-Line Blocking: FIFO vs Short-First', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_arrival_comparison(burst_results: Dict, poisson_results: Dict, save_path: str):
    """Plot arrival process comparison."""
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Queue length over time (placeholder - would need time series data)
    # For now, just show summary bars
    
    x = ['Burst', 'Poisson']
    width = 0.35
    
    # TTFT
    ttft_p99 = [burst_results["ttft_p99"], poisson_results["ttft_p99"]]
    e2e_p99 = [burst_results["e2e_p99"], poisson_results["e2e_p99"]]
    
    axes[0].bar([0 - width/2, 1 - width/2], ttft_p99, width, label='TTFT P99', color='#3498db')
    axes[0].bar([0 + width/2, 1 + width/2], e2e_p99, width, label='E2E P99', color='#e74c3c')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(x)
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Latency Comparison')
    axes[0].legend()
    
    # Throughput
    throughput = [burst_results["throughput_tok_s"], poisson_results["throughput_tok_s"]]
    axes[1].bar(x, throughput, color='steelblue')
    axes[1].set_ylabel('Tokens/sec')
    axes[1].set_title('Throughput')
    
    # Summary table as text
    axes[2].axis('off')
    table_data = [
        ['Metric', 'Burst', 'Poisson'],
        ['Tok/s', f'{burst_results["throughput_tok_s"]:.1f}', f'{poisson_results["throughput_tok_s"]:.1f}'],
        ['TTFT P99', f'{burst_results["ttft_p99"]:.0f}ms', f'{poisson_results["ttft_p99"]:.0f}ms'],
        ['E2E P99', f'{burst_results["e2e_p99"]:.0f}ms', f'{poisson_results["e2e_p99"]:.0f}ms'],
    ]
    axes[2].table(cellText=table_data[1:], colLabels=table_data[0], cellLoc='center', loc='center')
    axes[2].set_title('Summary')
    
    plt.suptitle('Arrival Process: Burst vs Poisson', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_regime_sweep(sweep_a: List[Dict], sweep_b: List[Dict], save_path: str):
    """Plot regime sweep results."""
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sweep A: TTFT vs prompt length
    prompt_lens = [r["prompt_tokens"] for r in sweep_a]
    ttfts = [r["ttft_ms"] for r in sweep_a]
    axes[0, 0].plot(prompt_lens, ttfts, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Prompt Length (tokens)')
    axes[0, 0].set_ylabel('TTFT (ms)')
    axes[0, 0].set_title('Sweep A: TTFT vs Prompt Length')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sweep A: Breakdown
    decode_times = [r["total_ms"] - r["ttft_ms"] for r in sweep_a]
    axes[0, 1].bar(prompt_lens, ttfts, label='Prefill', color='#3498db')
    axes[0, 1].bar(prompt_lens, decode_times, bottom=ttfts, label='Decode', color='#e74c3c')
    axes[0, 1].set_xlabel('Prompt Length (tokens)')
    axes[0, 1].set_ylabel('Time (ms)')
    axes[0, 1].set_title('Sweep A: Time Breakdown')
    axes[0, 1].legend()
    
    # Sweep B: Per-token latency
    gen_lens = [r["gen_tokens"] for r in sweep_b]
    per_token = [r["per_token_ms"] for r in sweep_b]
    axes[1, 0].plot(gen_lens, per_token, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    axes[1, 0].set_xlabel('Generation Length (tokens)')
    axes[1, 0].set_ylabel('Per-Token Latency (ms)')
    axes[1, 0].set_title('Sweep B: Per-Token Latency vs Gen Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sweep B: Total time
    ttfts_b = [r["ttft_ms"] for r in sweep_b]
    decode_times_b = [r["total_ms"] - r["ttft_ms"] for r in sweep_b]
    axes[1, 1].bar(gen_lens, ttfts_b, label='Prefill', color='#3498db')
    axes[1, 1].bar(gen_lens, decode_times_b, bottom=ttfts_b, label='Decode', color='#e74c3c')
    axes[1, 1].set_xlabel('Generation Length (tokens)')
    axes[1, 1].set_ylabel('Time (ms)')
    axes[1, 1].set_title('Sweep B: Time Breakdown')
    axes[1, 1].legend()
    
    plt.suptitle('Prefill vs Decode Regime Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
