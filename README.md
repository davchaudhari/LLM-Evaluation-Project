# Modal Serving Alpha

Production-grade LLM serving systems evaluation framework comparing HuggingFace and vLLM with rigorous benchmarking, statistical validation, and scheduling interventions.

## Overview

This repository contains a comprehensive evaluation framework for LLM serving systems, featuring:

- **Direct Backend Benchmarks**: True streaming TTFT measurement via `AsyncLLMEngine`
- **Scheduler-Layer Experiments**: Length-aware microbatch scheduling interventions
- **Statistical Validation**: Mann-Whitney U tests, effect size analysis, and confidence intervals
- **Fail-Loudly Design**: Request count validation and comparability checks

## Key Results

### Intervention Experiment
- **93.4% improvement** in short-request E2E P99 latency (18,558.7 ms → 1,224.9 ms)
- **95.6% improvement** in short-request TTFT P99 (13,911.3 ms → 617.4 ms)
- **+76% throughput** increase (222.4 → 391.4 tok/s)
- Statistically validated with matched workloads (32 requests each)

### Direct Benchmark
- **vLLM 93× faster TTFT** than HuggingFace (47.4 ms vs 4,425.1 ms)
- **vLLM 33× higher throughput** than HuggingFace sequential (870.7 vs 26.0 tok/s)
- True streaming measurement via `AsyncLLMEngine`

See [RESULTS_OUTPUT.md](RESULTS_OUTPUT.md) and [REVIEW_AND_RESULTS.md](REVIEW_AND_RESULTS.md) for detailed results and methodology review.

## Quick Start

### Run All Experiments

```bash
# HuggingFace baseline
modal run modal_app.py::run_hf_suite

# vLLM baseline (true continuous batching)
modal run modal_app.py::run_vllm_suite

# Intervention (length-aware + microbatch)
modal run modal_app.py::run_intervention_suite

# Direct benchmark (true streaming TTFT)
modal run modal_app.py::run_direct_benchmark

# Profiling
modal run modal_app.py::profile_vllm
```

### Generate Plots and Report

```bash
modal run modal_app.py::generate_report
```

### View Results Summary

```bash
modal run modal_app.py::show_results_summary
```

## Experiment Suite

1. **Direct Benchmark**: True backend performance with streaming TTFT measurement
2. **Intervention Suite**: Length-aware scheduling vs FIFO baseline
3. **TTFT Stair-Step**: N=16, batch=8, shows queueing effect
4. **Dispatch Policies**: fill_batch, periodic, max_wait, short_first
5. **Head-of-Line Blocking**: FIFO vs Short-first with mixed lengths
6. **Arrival Process**: Burst vs Poisson
7. **Regime Sweep**: Prefill-heavy vs Decode-heavy

## Results

All results saved to Modal volume `/results/`:
- `results/runs/`: JSONL logs per experiment
- `results/profiles/`: Profiler traces
- `results/figures/`: Plots
- `results/direct_benchmark/`: Direct benchmark results
- `results/intervention_results.json`: Intervention experiment summary

## Methodology

### Measurement Layers
- **`direct_backend`**: Bypasses scheduler, true streaming TTFT via AsyncLLMEngine
- **`scheduler_layer`**: End-to-end system performance, approximate TTFT (batch start time)

### Validation
- Request count matching (expected vs processed)
- Comparability checks (refuses invalid comparisons)
- Timing order validation (arrival ≤ first_token ≤ end)
- Statistical significance testing (Mann-Whitney U, effect size)

## Requirements

- Modal account
- GPU access (T4 minimum, A10G/L4 preferred)
- Python 3.10+
- See `pyproject.toml` for dependencies

## Structure

```
modal-serving-alpha/
├── modal_app.py              # Main Modal app with experiment functions
├── src/
│   ├── backends/             # HuggingFace and vLLM backend implementations
│   ├── benchmarks/           # Direct benchmark (true streaming TTFT)
│   ├── experiments/          # Experiment runner, metrics, workloads
│   ├── serving/              # Scheduler and dispatch policies
│   └── utils/                # I/O, timing, validation utilities
├── RESULTS_OUTPUT.md         # Detailed experimental results
├── REVIEW_AND_RESULTS.md     # Code review and merge assessment
└── audit_report.md           # Comprehensive audit report
```

## License

See LICENSE file for details.
