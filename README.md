# LLM-Evaluation-Project

Production-grade LLM serving systems evaluation framework comparing HuggingFace and vLLM with rigorous benchmarking, statistical validation, and scheduling interventions.

## Overview

This repository contains a comprehensive evaluation framework for LLM serving systems, featuring:

- **Direct Backend Benchmarks**: True streaming TTFT measurement via `AsyncLLMEngine`
- **Scheduler-Layer Experiments**: Length-aware microbatch scheduling interventions
- **Statistical Validation**: Mann-Whitney U tests, effect size analysis, and confidence intervals
- **Fail-Loudly Design**: Request count validation and comparability checks

## Key Results

### Intervention Experiment (scheduler-layer)
- **93.5% improvement** in short-request E2E P99 latency (524,870.1 ms → 34,206.0 ms)
- **93.5% improvement** in short-request TTFT P99 (520,614.9 ms → 33,654.9 ms)
- **+75.6% throughput** increase (241.3 → 423.8 tok/s)
- Statistically validated with matched workloads (1,000 requests each: 500 short + 500 long)

### Direct Benchmark
- **vLLM 61× lower TTFT P99** than HuggingFace (49.2 ms vs 3,006.9 ms)
- **vLLM 29× higher throughput** than HuggingFace sequential (930.9 vs 31.9 tok/s, vLLM concurrent)
- True streaming measurement via `AsyncLLMEngine`

See [RESULTS_OUTPUT.md](RESULTS_OUTPUT.md) for detailed experimental results. This file is regenerated from the latest Modal run via `modal run modal_app.py::generate_report` (which writes `/results/report.md` on the Modal volume).

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
└── RESULTS_OUTPUT.md         # Detailed experimental results (generated)
```

## License

See LICENSE file for details.
