#!/usr/bin/env python3
"""Create unified direct benchmark results file.

Merges hf_results.json and vllm_results.json into results.json
for a complete direct benchmark comparison.
"""

import json
from pathlib import Path


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_unified_results():
    """Create unified results.json from component files."""
    results_dir = Path("/results/direct_benchmark")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Load component results
    hf_path = results_dir / "hf_results.json"
    vllm_path = results_dir / "vllm_results.json"
    
    if not hf_path.exists():
        print(f"⚠️  HF results not found: {hf_path}")
        hf_data = None
    else:
        hf_data = load_json(hf_path)
        print(f"✓ Loaded HF results from {hf_path}")
    
    if not vllm_path.exists():
        print(f"⚠️  vLLM results not found: {vllm_path}")
        vllm_data = None
    else:
        vllm_data = load_json(vllm_path)
        print(f"✓ Loaded vLLM results from {vllm_path}")
    
    # Create unified structure
    unified = {
        "experiment_type": "direct_benchmark",
        "measurement_layer": "direct_backend",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "timestamp": "2026-01-06",
        "workload": {
            "hf_sequential": {
                "num_requests": 16,
                "prompt_tokens": 128,
                "max_new_tokens": 64,
            },
            "vllm_sequential": {
                "num_requests": 16,
                "prompt_tokens": 128,
                "max_new_tokens": 64,
            },
            "vllm_concurrent": {
                "num_requests": 32,
                "prompt_tokens": 128,
                "max_new_tokens": 64,
                "max_concurrent": 16,
            },
        },
        "results": {
            "huggingface_sequential": hf_data,
            "vllm_sequential": vllm_data.get("vllm_streaming_sequential") if vllm_data else None,
            "vllm_concurrent": vllm_data.get("vllm_concurrent") if vllm_data else None,
            "mixed_workload": vllm_data.get("mixed") if vllm_data else None,
        },
        "comparisons": {},
    }
    
    # Add comparisons if data available
    if hf_data and vllm_data:
        vllm_seq = vllm_data.get("vllm_streaming_sequential", {})
        vllm_conc = vllm_data.get("vllm_concurrent", {})
        
        # Sequential comparison (apples-to-apples)
        if vllm_seq:
            unified["comparisons"]["sequential"] = {
                "throughput_ratio": vllm_seq.get("throughput_tok_s", 0) / hf_data.get("throughput_tok_s", 1),
                "ttft_p99_ratio": hf_data.get("ttft_p99_ms", 0) / vllm_seq.get("ttft_p99_ms", 1),
                "e2e_p99_ratio": hf_data.get("e2e_p99_ms", 0) / vllm_seq.get("e2e_p99_ms", 1),
            }
        
        # Concurrent (note: different request counts)
        if vllm_conc:
            unified["comparisons"]["concurrent"] = {
                "note": "Different request counts: vLLM 32, HF 16",
                "throughput_ratio": vllm_conc.get("throughput_tok_s", 0) / hf_data.get("throughput_tok_s", 1),
                "ttft_p99_ratio": hf_data.get("ttft_p99_ms", 0) / vllm_conc.get("ttft_p99_ms", 1),
            }
    
    # Save unified results
    output_path = results_dir / "results.json"
    with open(output_path, 'w') as f:
        json.dump(unified, f, indent=2)
    
    print(f"\n✓ Unified results saved to {output_path}")
    print(f"  - HF Sequential: {'✓' if hf_data else '✗'}")
    print(f"  - vLLM Sequential: {'✓' if vllm_data and 'vllm_streaming_sequential' in vllm_data else '✗'}")
    print(f"  - vLLM Concurrent: {'✓' if vllm_data and 'vllm_concurrent' in vllm_data else '✗'}")
    print(f"  - Mixed Workload: {'✓' if vllm_data and 'mixed' in vllm_data else '✗'}")
    
    return unified


if __name__ == "__main__":
    create_unified_results()
