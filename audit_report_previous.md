# Modal Serving Alpha - Audit Report

## Summary Verdict: SHIP WITH MINOR CAVEATS

**Confidence Level: 85%** that results would hold under reviewer scrutiny.

~~The project has solid foundations (direct benchmark infrastructure, streaming TTFT measurement), but contains a **critical bug** that invalidates the headline intervention claim. The 96.8% improvement figure is an artifact of incomplete request processing, not a real scheduling improvement.~~

**UPDATE (2026-01-06)**: Critical bug has been FIXED and validated. The intervention now processes all 32 requests correctly, yielding a **validated 93.4% improvement**.

---

## 1. METRIC CORRECTNESS AUDIT

### 1.1 TTFT Computation

**Location**: `src/experiments/metrics.py` (lines 44-52)

```python
@property
def ttft_ms(self) -> float:
    if self.first_token_time is not None and self.arrival_time is not None:
        return (self.first_token_time - self.arrival_time) * 1000
    return 0.0
```

**Verdict: ✅ CORRECT (after recent fix)**

- Uses `is not None` checks, not truthiness
- Units: timestamps in seconds, output in milliseconds
- Formula: `(first_token_time - arrival_time) * 1000`

### 1.2 E2E Computation

**Location**: `src/experiments/metrics.py` (lines 54-59)

**Verdict: ✅ CORRECT**
- Same pattern as TTFT
- Formula: `(end_time - arrival_time) * 1000`

### 1.3 Throughput Computation

**Location**: `src/experiments/metrics.py` (lines 134-139)

```python
"throughput_tok_s": total_tokens / total_time if total_time > 0 else 0.0,
```

**Verdict: ✅ CORRECT**
- Uses wall-clock time from min(arrival) to max(end)
- Total tokens summed from all requests

### 1.4 Percentile Computation

**Location**: `src/experiments/metrics.py` (lines 141-145)

```python
def percentile(data, p):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]
```

**Verdict: ⚠️ MINOR ISSUE**
- Uses per-request arrays ✅
- Returns 0.0 for empty arrays (could mask bugs)
- Index calculation is approximate (should use `len * p / 100 - 1` for proper quantile)

### 1.5 Invariant Enforcement

**Location**: `src/experiments/metrics.py` (lines 61-89)

**Verdict: ✅ CORRECTLY IMPLEMENTED**
- Checks: `arrival <= first_token <= end`
- Checks: `e2e_ms >= ttft_ms`
- Detects timing order violations

### 1.6 Direct Benchmark TTFT

**Location**: `src/benchmarks/streaming_benchmark.py` (lines 70-73)

```python
async for output in self.engine.generate(...):
    if first_token_time is None and output.outputs[0].token_ids:
        first_token_time = time.perf_counter()
```

**Verdict: ✅ TRUE STREAMING TTFT**
- Captures actual first token arrival time
- Uses `time.perf_counter()` for precision

---

## 2. WORKLOAD CONSISTENCY CHECK

### 2.1 HF vs vLLM Direct Benchmark

**Location**: `modal_app.py::run_direct_benchmark()` (lines 1666-1685)

| Parameter | HF | vLLM | Match? |
|-----------|-----|------|--------|
| model_id | Qwen/Qwen2.5-3B-Instruct | Qwen/Qwen2.5-3B-Instruct | ✅ |
| num_requests | 16 (subset) | 16 (sequential) / 32 (concurrent) | ⚠️ |
| prompt_tokens | 128 | 128 | ✅ |
| max_new_tokens | 64 | 64 | ✅ |

**Verdict: ⚠️ MINOR MISMATCH**
- HF runs 16 requests, vLLM concurrent runs 32
- This affects throughput comparison but not per-request latency metrics
- Explicitly documented in code: "Subset for HF (slow)"

### 2.2 FIFO vs Length-Aware Intervention

**Location**: `modal_app.py::run_intervention_suite()` (lines 1011-1042)

| Parameter | FIFO | Intervention | Match? |
|-----------|------|--------------|--------|
| model_id | Qwen/Qwen2.5-3B-Instruct | Qwen/Qwen2.5-3B-Instruct | ✅ |
| num_requests | 32 | 32 (workload) | ✅ |
| short_tokens | 32 | 32 | ✅ |
| long_tokens | 256 | 256 | ✅ |
| short_ratio | 0.5 | 0.5 | ✅ |
| max_batch_size | 8 | 8 | ✅ |
| backend | vllm | vllm | ✅ |

**Verdict: ❌ CRITICAL MISMATCH IN ACTUAL EXECUTION**

From run output:
- **FIFO processed: 32 requests**
- **Intervention processed: 8 requests**

The intervention only processed 8 requests (one batch) while FIFO processed all 32. See Section 3 for root cause.

---

## 3. INTERVENTION RESULT SANITY CHECK

### 3.1 The Claim

> "96.8% improvement in short-request E2E P99"
> - Baseline (FIFO): 18,744.9 ms
> - Intervention: 604.1 ms

### 3.2 Critical Bug Found and FIXED

**Root Cause**: `LengthAwareMicrobatchPolicy` stored requests in internal buckets (`self.buckets`), but the main loop in `ExperimentRunner.run_workload()` only checked `pending` and `active_requests`.

**Fix Applied**:
1. Added `has_pending()` method to `DispatchPolicy` base class (returns False by default)
2. Implemented `has_pending()` in `LengthAwareMicrobatchPolicy` to check internal buckets
3. Modified loop condition: `while pending or active_requests or self.policy.has_pending()`
4. Added request count validation that raises `RuntimeError` if mismatch detected

### 3.3 Validated Results (After Fix)

| Metric | FIFO (32 req) | Intervention (32 req) | Valid? |
|--------|---------------|----------------------|--------|
| Total Requests | 32 | 32 | ✅ YES |
| E2E P99 (all) | 18,558.7 ms | 10,547.5 ms | ✅ YES |
| E2E P99 (short) | 18,558.7 ms | 1,224.9 ms | ✅ YES |
| Throughput | 222.4 tok/s | 391.4 tok/s | ✅ YES |
| Improvement | - | **93.4%** | ✅ VALID |

### 3.4 Verdict

**✅ THE 93.4% IMPROVEMENT CLAIM IS NOW VALID**

Both baseline and intervention processed identical workloads (32 requests, 16 short + 16 long).
The improvement is real and represents reduced head-of-line blocking for short requests.

---

## 4. LAYER SEPARATION CHECK

### 4.1 Code Labels

| Component | measurement_layer | Correct? |
|-----------|------------------|----------|
| `MetricsAggregator` default | "scheduler_layer" | ✅ |
| `ExperimentRunner` | "scheduler_layer" | ✅ |
| `DirectVLLMBenchmark` | (not tagged) | ⚠️ |
| `DirectHFBenchmark` | (not tagged) | ⚠️ |

**Issue**: Direct benchmarks don't use the `measurement_layer` field, which is only in `RequestMetrics`. The direct benchmarks use a separate `BenchmarkResult` class.

### 4.2 Report Layer Confusion

**EXPERIMENT_RESULTS.md line 100-105**:
```markdown
| Metric | Baseline (FIFO) | Intervention | Improvement |
|--------|-----------------|--------------|-------------|
| Throughput | 220.2 tok/s | 423.8 tok/s | +92% |
```

**Issue**: Throughput comparison is invalid because request counts differ.

**EXPERIMENT_RESULTS.md line 19-24** (Direct Benchmark):
```markdown
### vLLM Sequential (Streaming)
| Metric | Value | vs HF |
| Throughput | 61.7 tok/s | 2.4x faster |
```

**Issue**: HF uses 16 requests, vLLM sequential uses 16, concurrent uses 32. Throughput ratios may be misleading.

### 4.3 Verdict

**⚠️ PARTIAL COMPLIANCE**
- Layers are conceptually separated
- Documentation could conflate scheduler-layer intervention with direct backend claims

---

## 5. HF vs vLLM FAIRNESS CHECK

### 5.1 HF TTFT = E2E Claim

**Code verification** (`direct_benchmark.py` line 319):
```python
first_token_time=end_time,  # HF: TTFT = E2E (no streaming)
```

**Verdict: ✅ CORRECTLY IMPLEMENTED**

HuggingFace uses blocking `model.generate()` with no token callback. Users observe complete output only at end, so TTFT = E2E is semantically correct for user experience.

### 5.2 HF Streaming Availability

**Code**: `hf_backend.py` has `generate_stream()` method, but:
```python
async def generate_stream(self, request: GenerationRequest):
    # For HF, we'd need TextIteratorStreamer
    # For now, just return full result
    results = self.generate([request])
```

**Verdict: ✅ FAIR COMPARISON**

HF streaming exists but is documented as not truly streaming. The benchmark correctly uses blocking `generate()` to represent typical HF usage.

---

## 6. QSERVE SPIKE REVIEW

### 6.1 Failure Classification

**Error**: `python setup.py develop did not run successfully` (exit code 1)

**Root cause hypothesis** (from `scripts/qserve/notes.md`):
1. Missing CUDA architecture specification for A10G (sm_86)
2. cutlass submodule not initialized
3. PyTorch/CUDA version mismatch

**Fix attempted**: `TORCH_CUDA_ARCH_LIST="8.6"` - failed

### 6.2 Verdict

**Classification: MEDIUM-EFFORT FIX**

The error is plausible (CUDA kernel compilation is notoriously environment-sensitive). The notes.md documents the failure adequately. A medium-effort fix would involve:
- Full build log capture (currently only partial)
- Testing with QServe's recommended Docker image
- Trying pinned PyTorch/CUDA versions from QServe docs

**Reasonable to stop**: Yes, given the timeboxed nature of the spike.

---

## 7. CLAIMS VS EVIDENCE MAP

| Claim | Evidence File | Metric | Valid? |
|-------|--------------|--------|--------|
| "vLLM 33x faster throughput than HF" | direct_benchmark/results.json | 870.7 vs 26.0 tok/s | ⚠️ Different request counts |
| "93x faster TTFT" | direct_benchmark/vllm_results.json | 47.4 vs 4425.1 ms | ✅ Valid (P99) |
| "93.4% improvement in short E2E P99" | intervention_results.json | 18558.7 → 1224.9 ms | ✅ VALID (32 vs 32 requests) |
| "HF TTFT ≈ E2E due to non-streaming" | direct_benchmark.py code | first_token_time=end_time | ✅ Code confirms |
| "QServe kernel build failed" | qserve_build_logs/*.json | exit code 1 | ✅ Documented |

---

## Confirmed Strengths

1. **Direct benchmark infrastructure is solid** - True streaming TTFT measurement via AsyncLLMEngine
2. **Truthiness bug in metrics was fixed** - `is not None` checks are correct
3. **Validation/invariants are implemented** - Would catch timing order violations
4. **Layer separation is conceptually clear** - scheduler_layer vs direct_backend
5. **HF baseline is fair** - Correctly represents blocking generate() behavior
6. **QServe failure is documented** - Reasonable to stop given spike constraints
7. **✅ Intervention bug FIXED** - Now processes all requests correctly
8. **✅ Request count validation added** - Fails loudly on mismatch
9. **✅ Comparability check added** - Refuses to compute improvement if counts differ

## Confirmed Weaknesses (Minor)

1. ~~**❌ CRITICAL: Intervention only processed 8/32 requests**~~ **FIXED**
2. **⚠️ Direct benchmark request count mismatch** - HF:16, vLLM concurrent:32 (documented)
3. **⚠️ Throughput ratios may be inflated** - Different workload sizes (documented)

## Remaining Red Flags

1. ~~**The 96.8% figure must be retracted**~~ **FIXED: Now 93.4% (validated)**
2. ~~**Request count validation should fail loudly**~~ **FIXED: RuntimeError on mismatch**
3. ~~**TTFT P99 of 0.1ms for intervention**~~ **FIXED: Now 617.4ms (realistic)**

---

## Suggested Wording Fixes

### ~~Current (INVALID)~~ NOW VALID:
> "Length-aware scheduling achieves **93.4% improvement** in short-request E2E P99"

✅ This claim is now supported by validated data (32 requests each).

### Minor Clarification Suggested:
> "vLLM 33x faster throughput than HF"

### Suggested:
> "vLLM concurrent mode (32 requests) achieved 870 tok/s vs HF sequential (16 requests) at 26 tok/s. Direct per-request comparison shows vLLM streaming is 2.4x faster throughput."

---

## Fixes Applied ✅

1. **✅ Fixed `LengthAwareMicrobatchPolicy`**: Added `has_pending()` method
2. **✅ Fixed `ExperimentRunner` loop**: Now checks `self.policy.has_pending()`
3. **✅ Added request count validation**: Raises `RuntimeError` on mismatch
4. **✅ Added comparability check**: Refuses to compute improvement if counts differ
5. **✅ Re-ran intervention suite**: Results validated (93.4% improvement)
6. **✅ Updated documentation**: EXPERIMENT_RESULTS.md reflects correct numbers

---

## Appendix: Bug Fix Applied

### Before (Bug)
```python
# run_suite.py::run_workload
while pending or active_requests:  # <-- BUG: missing policy.has_pending()
    ...
# Loop exits when pending is empty, but policy.buckets has 24 requests!
```

### After (Fixed)
```python
# policies.py - Added has_pending() to base class
class DispatchPolicy:
    def has_pending(self) -> bool:
        return False  # Override in policies with internal buffers

# policies.py - LengthAwareMicrobatchPolicy
def has_pending(self) -> bool:
    return any(len(q) > 0 for q in self.buckets.values())

# run_suite.py - Fixed loop condition
while pending or active_requests or self.policy.has_pending():
    ...

# run_suite.py - Added validation
if processed_request_ids != expected_request_ids:
    raise RuntimeError(f"Request count mismatch! ...")
```

### Validation Output
```
✓ Saved baseline logs: /results/runs/intervention_fifo.jsonl
  Validation: 32 requests, valid=True

✓ Saved intervention logs: /results/runs/intervention_length_aware.jsonl
  Validation: 32 requests, valid=True

✓ Comparability check passed: 32 requests each

📊 Short Request E2E P99 Improvement: 93.4%
✅ SUCCESS: Met target of ≥50% improvement!
```

---

*Audit completed: 2026-01-06*
*Bug fixed and validated: 2026-01-06*
*Auditor: Senior Systems Engineer Review*
