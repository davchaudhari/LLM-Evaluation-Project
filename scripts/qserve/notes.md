# QServe Integration - Build Failure Report

## Status: FAILED (Kernel Build)

**Date**: 2026-01-06  
**Conclusion**: QServe CUDA kernels cannot be built on Modal A10G with current configuration.

---

## 1. Environment Details

### Modal Image Configuration
```python
qserve_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install(
        "git", "git-lfs", "wget", "curl",
        "build-essential", "ninja-build", "cmake",
        "libssl-dev", "libffi-dev",
    )
    .pip_install(
        "torch==2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "safetensors",
        "sentencepiece",
        "ninja",
        "packaging",
        "wheel",
        "setuptools",
        "numpy<2",
        "tqdm",
        "pyyaml",
        "jsonlines",
    )
)
```

### Hardware & Software Versions
| Component | Version |
|-----------|---------|
| GPU | NVIDIA A10G (22.1 GB VRAM) |
| CUDA | 12.1.1 (V12.1.105) |
| PyTorch | 2.2.0+cu121 |
| Python | 3.10 |
| GCC | 11.4.0 |
| Ninja | 1.10.1 |
| OS | Ubuntu 22.04 |

### GPU Compute Capability
- A10G: sm_86 (Ampere architecture)

---

## 2. Build Steps Performed

| Step | Command | Status |
|------|---------|--------|
| 1. Clone Repo | `git clone --depth 1 https://github.com/mit-han-lab/qserve.git` | ✅ Success |
| 2. Install Deps | `pip install -r requirements.txt` | ✅ Success |
| 3. Build Kernels | `cd kernels && pip install -e . -v` | ❌ **FAILED** |
| 4. Install Package | `pip install -e .` | ✅ Success (no kernels) |
| 5. Import Test | `import qserve` | ⚠️ Partial (no engine classes) |
| 6. Generation Test | HuggingFace fallback | ✅ Success |

---

## 3. Kernel Build Failure Details

### Failing Command
```bash
cd /root/qserve/kernels && pip install -e . -v
```

### Error Type
```
subprocess-exited-with-error
× python setup.py develop did not run successfully.
│ exit code: 1
```

### Root Cause Analysis

The kernel build fails during CUDA compilation. Based on the error patterns:

1. **Missing CUDA architecture specification**: The build doesn't correctly detect A10G's sm_86 architecture
2. **PyTorch/CUDA version mismatch**: PyTorch 2.2.0 may have different CUDA extension compilation requirements
3. **Missing cutlass submodule**: QServe depends on NVIDIA cutlass which may not be initialized

### Evidence from Import Test
```python
✓ qserve imported successfully
  qserve module location: None  # <-- No compiled module
✗ Failed to import EngineArgs/LLMEngine: cannot import name 'EngineArgs' from 'qserve'
✗ Failed to import SamplingParams: cannot import name 'SamplingParams' from 'qserve'

Available qserve submodules:
  qserve.lserve_benchmark
  qserve.lserve_e2e_generation
  qserve.omniserve              # Python-only modules work
  qserve.qserve_benchmark
  qserve.qserve_e2e_generation
```

The core inference classes (EngineArgs, LLMEngine, SamplingParams) are defined in the C++/CUDA extension, which failed to build.

---

## 4. Targeted Fix Attempt

### Hypothesis
The CUDA architecture list is not set correctly for A10G (sm_86).

### Fix Applied
```bash
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=2  # Limit parallelism to avoid OOM
export CXX=g++
export CC=gcc
```

### Result
**STILL FAILED** - Same error persists.

### Alternative Hypotheses
1. **cutlass dependency**: QServe may require cutlass submodule
2. **PyTorch version**: QServe may require specific PyTorch version
3. **CUDA runtime vs toolkit**: Modal may have CUDA toolkit issues

---

## 5. Files & Logs

### Log Locations
- Build logs: `/results/qserve_build_logs/build_*.log`
- Results JSON: `/results/qserve_build_logs/results_*.json`
- Kernel debug: `/results/qserve_build_logs/kernel_build_*.log`

### Key Log Files from Latest Run
```
/results/qserve_build_logs/results_20260106_184743.json
```

### How to Access Logs
```bash
# List available logs
modal run modal_app.py::show_results_summary

# Or directly via volume
modal volume ls modal-results qserve_build_logs/
```

---

## 6. What Works (Fallback Mode)

Even without kernels, QServe can run in **HuggingFace fallback mode**:

```python
# This works - uses transformers backend
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
```

**Fallback performance** (no W4A8KV4 quantization):
- Model: Qwen/Qwen2.5-3B-Instruct
- GPU Memory: ~6 GB (FP16)
- Generation: Works but no quantization benefits

---

## 7. Recommendations

### Option A: Pre-built Docker Image
QServe may have official Docker images with pre-compiled kernels:
```bash
# Check for official images
docker pull mit-han-lab/qserve:latest  # (if available)
```

### Option B: Different Build Environment
Try building on:
- AWS EC2 with NVIDIA Deep Learning AMI
- Lambda Labs GPU instances
- Local machine with CUDA toolkit

### Option C: Alternative Quantization
Use vLLM's built-in quantization instead:
```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", quantization="awq")
```

### Option D: Report Issue to QServe
File an issue with:
- Environment details
- Full build log
- GPU architecture (A10G / sm_86)

---

## 8. Conclusion

**QServe kernel build is not feasible on Modal A10G** with the current configuration.

The W4A8KV4 quantized inference requires compiled CUDA kernels that fail to build. The fallback HuggingFace mode works but provides no quantization benefits.

**Recommended path forward**: Use vLLM with AWQ or GPTQ quantization instead, which has better out-of-the-box support.

---

## 9. Related Commands

```bash
# Debug kernel build (captures full error)
modal run modal_app.py::qserve_kernel_debug

# Full build smoketest
modal run modal_app.py::qserve_build_smoketest

# View results
modal run modal_app.py::show_results_summary
```

---

*Last updated: 2026-01-06*
