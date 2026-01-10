"""Regime sweep experiments (prefill vs decode)."""

import torch
from typing import List, Dict
from ..backends.backend_base import BackendBase, GenerationRequest
from ..utils.tokenize import get_tokenizer


def generate_prompt(target_tokens: int, tokenizer) -> str:
    """Generate a prompt with approximately target_tokens tokens."""
    base = "Answer the following question in detail: What is the meaning of life? "
    tokens = tokenizer.encode(base, add_special_tokens=True)
    
    while len(tokens) < target_tokens:
        base += "Please elaborate further. "
        tokens = tokenizer.encode(base, add_special_tokens=True)
    
    # Truncate to exact length
    tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens)


def measure_generation(
    backend: BackendBase,
    prompt: str,
    max_new_tokens: int,
    tokenizer,
    num_runs: int = 3
) -> Dict:
    """Measure TTFT and per-token latency."""
    import time
    
    ttfts = []
    per_token_latencies = []
    total_times = []
    
    for _ in range(num_runs):
        req = GenerationRequest(
            request_id=f"regime_test_{_}",
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        
        # Measure prefill (TTFT)
        start = time.perf_counter()
        results = backend.generate([req])
        end = time.perf_counter()
        
        # For now, approximate TTFT as total time
        # In production, would use streaming for accurate measurement
        ttft = (end - start) * 1000
        total_time = ttft
        
        ttfts.append(ttft)
        total_times.append(total_time)
        
        # Approximate per-token (decode time / tokens)
        if results and results[0].generated_tokens > 1:
            # Rough estimate: assume decode is most of the time
            decode_time = total_time * 0.9  # Approximate
            per_token = decode_time / results[0].generated_tokens
            per_token_latencies.append(per_token)
    
    return {
        "ttft_ms": sum(ttfts) / len(ttfts),
        "per_token_ms": sum(per_token_latencies) / len(per_token_latencies) if per_token_latencies else 0,
        "total_ms": sum(total_times) / len(total_times),
    }


def run_prefill_sweep(
    backend: BackendBase,
    model_id: str,
    prompt_lengths: List[int],
    fixed_gen_tokens: int = 32
) -> List[Dict]:
    """Sweep prompt length (prefill-heavy)."""
    tokenizer = get_tokenizer(model_id)
    results = []
    
    for prompt_len in prompt_lengths:
        prompt = generate_prompt(prompt_len, tokenizer)
        result = measure_generation(backend, prompt, fixed_gen_tokens, tokenizer)
        result["prompt_tokens"] = prompt_len
        result["gen_tokens"] = fixed_gen_tokens
        results.append(result)
    
    return results


def run_decode_sweep(
    backend: BackendBase,
    model_id: str,
    gen_lengths: List[int],
    fixed_prompt_tokens: int = 64
) -> List[Dict]:
    """Sweep generation length (decode-heavy)."""
    tokenizer = get_tokenizer(model_id)
    prompt = generate_prompt(fixed_prompt_tokens, tokenizer)
    results = []
    
    for gen_len in gen_lengths:
        result = measure_generation(backend, prompt, gen_len, tokenizer)
        result["prompt_tokens"] = fixed_prompt_tokens
        result["gen_tokens"] = gen_len
        results.append(result)
    
    return results
