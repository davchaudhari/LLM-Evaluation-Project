# LLM Serving Experiments Report

## Executive Summary

This report compares HuggingFace Transformers (sequential processing) with vLLM (continuous batching) across a direct backend benchmark, dispatch policies, head-of-line blocking, arrival process, and a scheduling intervention.

**Important — arrival model:** Except where noted, the `scheduler_layer` suite
experiments (Sections 1–4) submit all requests as a synchronized burst
(arrival_time = 0). Under a burst, large latency gaps partly reflect queueing of
a fixed backlog rather than steady-state serving behavior, so the percentage
gaps in those sections should be read as burst-queueing comparisons, not
open-loop serving wins. The Scheduling Intervention (Section 5) uses a realistic
Poisson open-loop arrival process and is the load-realistic result; the Direct
Benchmark (Section 0) bypasses the scheduler entirely.

**Reading this report:** All numbers below are produced directly from the saved
result files for the latest run; this summary contains no hand-entered figures.
Two measurement layers are reported separately — `direct_backend` (true
per-request streaming TTFT) and `scheduler_layer` (system-level E2E with
approximate, batch-start TTFT). Compare like-for-like within a layer.

## 0. Direct Backend Benchmark (scheduler bypassed, true streaming TTFT)

| Config | Throughput (tok/s) | TTFT P50 (ms) | TTFT P99 (ms) | E2E P99 (ms) | N |
| --- | --- | --- | --- | --- | --- |
| HF sequential | 31.9 | 1930.2 | 3006.9 | 3006.9 | 16 |
| vLLM sequential | 64.8 | 23.2 | 25.8 | 988.2 | 16 |
| vLLM concurrent | 930.9 | 43.8 | 49.2 | 1098.9 | 32 |

vLLM concurrent throughput is **29.2x** HuggingFace sequential (930.9 vs 31.9 tok/s).

Note: the HuggingFace path is non-streaming, so its measured TTFT equals its E2E latency; vLLM TTFT is true per-request streaming via `AsyncLLMEngine`.

## 1. Backend Comparison: HF vs vLLM (scheduler-layer, burst arrivals)

### Key Differences

- **HuggingFace**: Sequential processing, no true batching, GPU underutilized
- **vLLM**: Continuous batching with PagedAttention, token-level interleaving, high GPU utilization

### Throughput Comparison


| Backend | Throughput (tok/s) | TTFT P99 (ms) | E2E P99 (ms) |
|---------|-------------------|---------------|--------------|
| HuggingFace | 21.5 | 20352 | 44372 |
| vLLM | 483.9 | 0 | 1038 |

**Throughput improvement:** 2150.2%


### Background: vLLM continuous batching

These are design properties of vLLM (not claims derived from this report's
tables); the throughput figures above are the measured evidence:
1. **Continuous batching**: new requests join the running batch at token
   boundaries rather than waiting for the whole batch to finish.
2. **Token-level interleaving**: multiple requests decode concurrently.
3. **PagedAttention**: paged KV-cache management enables larger effective batches.

## 2. Dispatch Policy Analysis

| Policy | Tok/s | TTFT P50 | TTFT P99 | E2E P99 |
| --- | --- | --- | --- | --- |
| fill_batch | 483.9 | 0 | 0 | 1038 |
| periodic_5ms | 484.0 | 0 | 0 | 1047 |
| periodic_10ms | 483.4 | 0 | 0 | 1050 |
| periodic_20ms | 483.7 | 0 | 0 | 1052 |
| periodic_50ms | 484.0 | 0 | 0 | 1059 |
| max_wait_50ms | 484.4 | 0 | 0 | 1055 |
| max_wait_100ms | 484.5 | 0 | 0 | 1055 |
| max_wait_200ms | 484.3 | 0 | 0 | 1042 |
| short_first | 484.1 | 0 | 0 | 1050 |


**Note:** These are scheduler-layer measurements. TTFT at this layer is
approximate (batch-start time), so TTFT percentiles can read as 0 when the
batch starts in the same tick a request is dispatched. Treat throughput and
E2E latency as the reliable signals here; for true per-request TTFT see the
Direct Benchmark section.

## 3. Head-of-Line Blocking (scheduler-layer, burst arrivals)


| Policy | Short E2E P99 (ms) | Long E2E P99 (ms) | Improvement |
|--------|-------------------|------------------|-------------|
| FIFO | 16646 | 16646 | - |
| Short-first | 1085 | 9450 | 93.5% |

**Finding:** The Improvement column reports the measured short-request E2E P99
reduction of Short-first vs FIFO; long-request E2E P99 is shown alongside so
any regression on long requests is visible.

**Caveat:** This experiment submits all requests as a synchronized burst
(arrival_time = 0). The large improvement is therefore dominated by re-ordering
a fixed backlog and is *not* a steady-state serving result — note FIFO short and
long E2E P99 are identical, the signature of a burst backlog. For the
load-realistic version of the same length-aware idea under an open-loop Poisson
process, see Section 5, where the effect is small (~8%) and not statistically
significant.


## 4. Arrival Process Impact


| Mode | Throughput (tok/s) | TTFT P99 (ms) | E2E P99 (ms) |
|------|-------------------|---------------|--------------|
| Burst | 482.3 | 3185 | 4246 |
| Poisson | 482.0 | 0 | 1056 |

**Finding:** Measured throughput and tail latency for burst vs Poisson
arrivals are shown in the table above.


## 5. Scheduling Intervention


**Length-Aware Microbatch Policy** (scheduler-layer; arrivals: Poisson)

| Metric | Baseline (FIFO) | Intervention | Change |
|--------|----------------|--------------|--------|
| Short E2E P99 | 8842.7 ms | 8123.0 ms | 8.1% lower |
| Short approx. TTFT P99 | 4741.3 ms | 3957.9 ms | — |
| Throughput | 139.4 tok/s | 142.4 tok/s | +2.2% |

Sample size: 300 requests (150 short + 150 long), matched across both arms.

**Statistical test (short-request E2E):** Mann-Whitney U = 11793, p = 2.35e-01, Cohen's d = 0.09.

**How it works:**
- Requests bucketed by max_new_tokens (short/medium/long)
- Microbatch window (20ms) allows batching within buckets
- Short requests get priority, reducing head-of-line blocking from long requests


## 6. Profiling Analysis


**Profiling Summary (host-process `torch.profiler`):**

| Metric | Value |
|--------|-------|
| Total CUDA Time | 0.0 ms |
| Total CPU Time | 256.1 ms |
| Max Memory (PyTorch allocator) | 0.0 MB |

**Limitation:** vLLM executes the model in a separate worker process (`EngineCore`),
so the host-side `torch.profiler` used here does not observe the GPU kernels or
device memory of that worker — hence CUDA time and max memory above are typically
0. These numbers therefore reflect only the host/orchestration process, not the
model's device-side execution. A kernel-level breakdown would require profiling
inside the vLLM worker (e.g. Nsight Systems or an in-worker profiler hook), which
is listed under Next Steps.


## 7. Limitations and Next Steps

### Current Limitations

1. **QServe Integration**: Not yet implemented (requires CUDA kernel build)
2. **Nsight Systems Profiling**: Not included (requires system-level access)
3. **Production Server**: HTTP API exists but not demonstrated under load
4. **Real-world Workloads**: Experiments use synthetic workloads

### Next Steps

1. **QServe Integration** (if build feasible on Modal):
   - Build QServe CUDA kernels
   - Compare W4A8KV4 quantized vs FP16 baseline
   - Measure memory reduction and throughput gains

2. **Nsight Systems Profiling**:
   - Capture kernel-level traces
   - Identify specific bottlenecks (attention, GEMM, memory)
   - Optimize based on profiling data

3. **Production Deployment**:
   - Deploy HTTP server on Modal
   - Run load tests with real concurrent clients
   - Measure production metrics (P95/P99 latencies)

4. **Advanced Scheduling**:
   - Implement two-queue system
   - Add deadline-based scheduling
   - Explore preemption strategies

## Reproducibility

All experiments can be reproduced with:

```bash
# HuggingFace baseline
modal run modal_app.py::run_hf_suite

# vLLM baseline
modal run modal_app.py::run_vllm_suite

# Intervention experiment
modal run modal_app.py::run_intervention_suite

# Profiling
modal run modal_app.py::profile_vllm

# Generate report
modal run modal_app.py::generate_report
```

**Model:** Qwen/Qwen2.5-3B-Instruct  
**GPU:** T4 (16GB) or A10G (24GB)  
**Batch Size:** 8  
**Seed:** 42 (deterministic workloads)

## Appendix: Full Results

See `/results/` directory for:
- JSONL logs per experiment
- JSON summaries
- PNG plots (when generated via `generate_plots`)
- Host-process profiler summary (`profiles/vllm_profile.json`)

