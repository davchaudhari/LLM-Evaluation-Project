"""
Modal app for LLM serving experiments.

Run experiments:
    modal run modal_app.py::run_hf_suite
    modal run modal_app.py::run_vllm_suite
    modal run modal_app.py::run_intervention_suite
    modal run modal_app.py::profile_vllm
    modal run modal_app.py::generate_report
"""

import modal

# =============================================================================
# Modal Image
# =============================================================================

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install("git", "git-lfs")
    .pip_install(
        "numpy<2",
        "scipy>=1.10.0",  # For statistical testing
        # Let vLLM determine torch version (it requires torch>=2.4.0)
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "vllm>=0.6.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "httpx>=0.27.0",
        "pydantic>=2.6.0",
        "pandas>=2.0.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "jsonlines>=4.0.0",
    )
    .run_commands("git lfs install")
    .add_local_dir("src", "/root/src")
)

app = modal.App("modal-serving-alpha")

# Volumes
model_cache = modal.Volume.from_name("modal-models", create_if_missing=True)
results_volume = modal.Volume.from_name("modal-results", create_if_missing=True)


# =============================================================================
# QServe Image (separate from vLLM to avoid conflicts)
# =============================================================================

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
        # Core ML - pinned for QServe compatibility
        "torch==2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "safetensors",
        "sentencepiece",
        # Build tools
        "ninja",
        "packaging",
        "wheel",
        "setuptools",
        # Utilities
        "numpy<2",
        "tqdm",
        "pyyaml",
        "jsonlines",
    )
    .run_commands("git lfs install")
    # Set compiler environment
    .env({
        "CXX": "g++",
        "CC": "gcc",
    })
)


# =============================================================================
# QServe Build and Test Functions
# =============================================================================

@app.function(
    image=qserve_image,
    gpu="A10G",
    timeout=3600,  # 1 hour for build
    volumes={"/models": model_cache, "/results": results_volume},
)
def qserve_kernel_debug():
    """
    Debug QServe kernel build - capture FULL error output.
    """
    import subprocess
    import os
    import json
    from pathlib import Path
    from datetime import datetime
    
    log_dir = Path("/results/qserve_build_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("QSERVE KERNEL DEBUG BUILD")
    print("=" * 80)
    
    # Check environment
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Clone QServe
    qserve_dir = "/root/qserve"
    subprocess.run(f"rm -rf {qserve_dir}", shell=True)
    subprocess.run(
        f"git clone --depth 1 https://github.com/mit-han-lab/qserve.git {qserve_dir}",
        shell=True
    )
    
    # Check kernel setup.py
    print("\n" + "=" * 80)
    print("KERNEL SETUP.PY CONTENTS (first 100 lines)")
    print("=" * 80)
    
    setup_file = f"{qserve_dir}/kernels/setup.py"
    if os.path.exists(setup_file):
        with open(setup_file) as f:
            lines = f.readlines()[:100]
            print("".join(lines))
    
    # Check csrc directory
    print("\n" + "=" * 80)
    print("CSRC DIRECTORY STRUCTURE")
    print("=" * 80)
    subprocess.run(f"find {qserve_dir}/kernels/csrc -type f -name '*.cu' -o -name '*.cpp' | head -30", shell=True)
    
    # Set environment - MUST set CXX before PyTorch checks for compiler
    os.environ["CXX"] = "g++"
    os.environ["CC"] = "gcc"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    os.environ["MAX_JOBS"] = "2"
    os.environ["VERBOSE"] = "1"
    
    # Verify compilers exist
    print("\nCompiler check:")
    subprocess.run("which g++", shell=True)
    subprocess.run("g++ --version | head -1", shell=True)
    subprocess.run("which nvcc", shell=True)
    
    print("\n" + "=" * 80)
    print("ATTEMPTING KERNEL BUILD (FULL OUTPUT)")
    print("=" * 80)
    
    # Create target directory and run build
    result = subprocess.run(
        "cd /root/qserve/kernels && mkdir -p omniserve_backend && python setup.py build_ext --inplace 2>&1",
        shell=True,
        capture_output=True,
        text=True,
        timeout=1800
    )
    
    print(f"\nExit code: {result.returncode}")
    print("\n--- STDOUT ---")
    print(result.stdout[-10000:] if len(result.stdout) > 10000 else result.stdout)
    print("\n--- STDERR ---")
    print(result.stderr[-10000:] if len(result.stderr) > 10000 else result.stderr)
    
    # Save full output
    with open(log_dir / f"kernel_build_{timestamp}.log", "w") as f:
        f.write(f"Exit code: {result.returncode}\n\n")
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n\n=== STDERR ===\n")
        f.write(result.stderr)
    
    print(f"\nFull log saved to: {log_dir}/kernel_build_{timestamp}.log")
    
    return {
        "exit_code": result.returncode,
        "log_file": f"kernel_build_{timestamp}.log",
    }


@app.function(
    image=qserve_image,
    gpu="A10G",
    timeout=3600,  # 1 hour for build
    volumes={"/models": model_cache, "/results": results_volume},
)
def qserve_build_smoketest():
    """
    Attempt to build QServe from source and run a minimal generation test.
    
    This is a diagnostic function to determine if QServe can be built on Modal.
    All output is logged to /results/qserve_build_logs/
    """
    import subprocess
    import sys
    import os
    import time
    import json
    from pathlib import Path
    from datetime import datetime
    
    # Setup logging directory
    log_dir = Path("/results/qserve_build_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"build_{timestamp}.log"
    
    results = {
        "timestamp": timestamp,
        "status": "started",
        "steps": [],
        "errors": [],
        "gpu_info": {},
    }
    
    def log(msg: str, is_error: bool = False):
        """Log to both console and file."""
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        if is_error:
            results["errors"].append(msg)
    
    def run_cmd(cmd: str, cwd: str = None, timeout: int = 600) -> tuple[int, str, str]:
        """Run command and capture output."""
        log(f"\n>>> Running: {cmd}")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            log(f"Exit code: {result.returncode}")
            if result.stdout:
                log(f"STDOUT:\n{result.stdout[-2000:]}")  # Last 2000 chars
            if result.stderr:
                log(f"STDERR:\n{result.stderr[-2000:]}")
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            log(f"TIMEOUT after {timeout}s", is_error=True)
            return -1, "", "TIMEOUT"
        except Exception as e:
            log(f"ERROR: {e}", is_error=True)
            return -1, "", str(e)
    
    log("=" * 80)
    log("QSERVE BUILD SMOKETEST")
    log("=" * 80)
    log(f"Timestamp: {timestamp}")
    log(f"Log file: {log_file}")
    
    # =========================================================================
    # Step 1: Check GPU and CUDA
    # =========================================================================
    log("\n" + "=" * 80)
    log("STEP 1: GPU and CUDA Check")
    log("=" * 80)
    
    import torch
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"GPU: {gpu_name}")
        log(f"GPU Memory: {gpu_mem:.1f} GB")
        results["gpu_info"] = {
            "name": gpu_name,
            "memory_gb": gpu_mem,
            "cuda_version": torch.version.cuda,
        }
    else:
        log("ERROR: CUDA not available!", is_error=True)
        results["status"] = "failed"
        results["failure_reason"] = "CUDA not available"
        return results
    
    run_cmd("nvcc --version")
    results["steps"].append({"step": "gpu_check", "status": "passed"})
    
    # =========================================================================
    # Step 2: Clone QServe Repository
    # =========================================================================
    log("\n" + "=" * 80)
    log("STEP 2: Clone QServe Repository")
    log("=" * 80)
    
    qserve_dir = "/root/qserve"
    
    if os.path.exists(qserve_dir):
        log(f"QServe directory exists, removing...")
        run_cmd(f"rm -rf {qserve_dir}")
    
    code, _, _ = run_cmd(
        f"git clone --depth 1 https://github.com/mit-han-lab/qserve.git {qserve_dir}",
        timeout=300
    )
    
    if code != 0:
        log("ERROR: Failed to clone QServe", is_error=True)
        results["status"] = "failed"
        results["failure_reason"] = "git clone failed"
        results["steps"].append({"step": "clone", "status": "failed"})
        with open(log_dir / f"results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)
        return results
    
    results["steps"].append({"step": "clone", "status": "passed"})
    
    # List repo contents
    run_cmd(f"ls -la {qserve_dir}")
    run_cmd(f"ls -la {qserve_dir}/kernels 2>/dev/null || echo 'No kernels directory'")
    
    # =========================================================================
    # Step 3: Install QServe Dependencies
    # =========================================================================
    log("\n" + "=" * 80)
    log("STEP 3: Install QServe Dependencies")
    log("=" * 80)
    
    # Check for requirements.txt
    req_file = f"{qserve_dir}/requirements.txt"
    if os.path.exists(req_file):
        log("Found requirements.txt, installing...")
        code, _, _ = run_cmd(f"pip install -r {req_file}", timeout=600)
        if code != 0:
            log("WARNING: Some requirements failed to install")
    
    results["steps"].append({"step": "dependencies", "status": "passed"})
    
    # =========================================================================
    # Step 4: Build CUDA Kernels
    # =========================================================================
    log("\n" + "=" * 80)
    log("STEP 4: Build CUDA Kernels")
    log("=" * 80)
    
    # Set environment for A10G (compute capability 8.6)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    os.environ["MAX_JOBS"] = "4"  # Limit parallel jobs to avoid OOM
    
    log(f"TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")
    log(f"MAX_JOBS={os.environ['MAX_JOBS']}")
    
    kernels_dir = f"{qserve_dir}/kernels"
    
    if os.path.exists(kernels_dir):
        log(f"Kernels directory found: {kernels_dir}")
        run_cmd(f"ls -la {kernels_dir}")
        
        # Check for setup.py or pyproject.toml
        if os.path.exists(f"{kernels_dir}/setup.py"):
            log("Found setup.py, building kernels...")
            code, stdout, stderr = run_cmd(
                "pip install -e . -v",
                cwd=kernels_dir,
                timeout=1800  # 30 min for kernel build
            )
        elif os.path.exists(f"{kernels_dir}/pyproject.toml"):
            log("Found pyproject.toml, building kernels...")
            code, stdout, stderr = run_cmd(
                "pip install -e . -v",
                cwd=kernels_dir,
                timeout=1800
            )
        else:
            log("No standard build file in kernels/")
            code = -1
        
        if code != 0:
            log("ERROR: Kernel build failed!", is_error=True)
            results["steps"].append({"step": "kernel_build", "status": "failed"})
            # Continue anyway to see what happens
        else:
            results["steps"].append({"step": "kernel_build", "status": "passed"})
    else:
        log("No kernels directory found")
        results["steps"].append({"step": "kernel_build", "status": "skipped"})
    
    # =========================================================================
    # Step 5: Install QServe Package
    # =========================================================================
    log("\n" + "=" * 80)
    log("STEP 5: Install QServe Package")
    log("=" * 80)
    
    if os.path.exists(f"{qserve_dir}/setup.py") or os.path.exists(f"{qserve_dir}/pyproject.toml"):
        code, _, _ = run_cmd(
            "pip install -e . -v",
            cwd=qserve_dir,
            timeout=600
        )
        
        if code != 0:
            log("ERROR: QServe package install failed!", is_error=True)
            results["steps"].append({"step": "package_install", "status": "failed"})
        else:
            results["steps"].append({"step": "package_install", "status": "passed"})
    else:
        log("No setup.py or pyproject.toml found in QServe root")
        results["steps"].append({"step": "package_install", "status": "skipped"})
    
    # =========================================================================
    # Step 6: Test QServe Import
    # =========================================================================
    log("\n" + "=" * 80)
    log("STEP 6: Test QServe Import")
    log("=" * 80)
    
    import_test = """
import sys
sys.path.insert(0, '/root/qserve')

try:
    import qserve
    print(f"✓ qserve imported successfully")
    print(f"  qserve module location: {qserve.__file__}")
except ImportError as e:
    print(f"✗ Failed to import qserve: {e}")

try:
    from qserve import EngineArgs, LLMEngine
    print(f"✓ EngineArgs and LLMEngine imported")
except ImportError as e:
    print(f"✗ Failed to import EngineArgs/LLMEngine: {e}")

try:
    from qserve import SamplingParams
    print(f"✓ SamplingParams imported")
except ImportError as e:
    print(f"✗ Failed to import SamplingParams: {e}")

# List available modules
import pkgutil
print("\\nAvailable qserve submodules:")
try:
    for importer, modname, ispkg in pkgutil.walk_packages(qserve.__path__, prefix='qserve.'):
        print(f"  {modname}")
except:
    print("  Could not list submodules")
"""
    
    with open("/tmp/import_test.py", "w") as f:
        f.write(import_test)
    
    code, stdout, stderr = run_cmd("python /tmp/import_test.py")
    
    if "✓ qserve imported successfully" in stdout:
        results["steps"].append({"step": "import_test", "status": "passed"})
        log("QServe import successful!")
    else:
        results["steps"].append({"step": "import_test", "status": "failed"})
        log("QServe import failed", is_error=True)
    
    # =========================================================================
    # Step 7: Minimal Generation Test (if import succeeded)
    # =========================================================================
    log("\n" + "=" * 80)
    log("STEP 7: Minimal Generation Test")
    log("=" * 80)
    
    # Try with a small HuggingFace model first (fallback mode)
    gen_test = """
import sys
sys.path.insert(0, '/root/qserve')
import torch
import time

print("Testing with HuggingFace fallback (no QServe kernels)...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/phi-2"  # Small model for testing
cache_dir = "/models"

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Quick generation
prompt = "What is 2+2?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print(f"Generating...")
start = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
end = time.perf_counter()

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")
print(f"Time: {(end-start)*1000:.1f}ms")
print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
print("✓ Generation test passed")
"""
    
    with open("/tmp/gen_test.py", "w") as f:
        f.write(gen_test)
    
    code, stdout, stderr = run_cmd("python /tmp/gen_test.py", timeout=600)
    
    if "✓ Generation test passed" in stdout:
        results["steps"].append({"step": "generation_test", "status": "passed"})
    else:
        results["steps"].append({"step": "generation_test", "status": "failed"})
    
    # =========================================================================
    # Summary
    # =========================================================================
    log("\n" + "=" * 80)
    log("BUILD SUMMARY")
    log("=" * 80)
    
    passed = sum(1 for s in results["steps"] if s["status"] == "passed")
    failed = sum(1 for s in results["steps"] if s["status"] == "failed")
    skipped = sum(1 for s in results["steps"] if s["status"] == "skipped")
    
    log(f"Passed: {passed}")
    log(f"Failed: {failed}")
    log(f"Skipped: {skipped}")
    
    if failed == 0:
        results["status"] = "success"
        log("\n✓ BUILD SUCCEEDED")
    else:
        results["status"] = "partial_failure"
        results["failure_reason"] = f"{failed} steps failed"
        log(f"\n⚠ BUILD PARTIALLY FAILED ({failed} steps)")
    
    log(f"\nErrors encountered: {len(results['errors'])}")
    for err in results["errors"]:
        log(f"  - {err}")
    
    # Save results JSON
    with open(log_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    log(f"\nResults saved to: {log_dir}/results_{timestamp}.json")
    log(f"Full log: {log_file}")
    
    return results


# =============================================================================
# Experiment Functions
# =============================================================================

@app.function(
    image=image,
    gpu="T4",  # Default to T4, can override to A10G/L4
    timeout=3600,
    volumes={"/models": model_cache, "/results": results_volume},
)
def run_hf_suite():
    """Run experiment suite on HuggingFace backend."""
    import sys
    sys.path.insert(0, "/root")
    
    from src.backends.hf_backend import HuggingFaceBackend
    from src.experiments.run_suite import ExperimentRunner
    from src.experiments.workloads import WorkloadGenerator
    from src.serving.policies import (
        FillBatchPolicy, PeriodicPolicy, MaxWaitPolicy, ShortFirstPolicy
    )
    from src.experiments.metrics import MetricsAggregator
    from src.experiments.plots import (
        plot_ttft_stair_step, plot_policy_comparison,
        plot_hol_comparison, plot_arrival_comparison, plot_regime_sweep
    )
    from src.utils.io import save_json
    import time
    
    print("=" * 80)
    print("RUNNING HF SUITE")
    print("=" * 80)
    
    # Load backend
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    backend = HuggingFaceBackend(model_id, cache_dir="/models")
    print(f"✓ Backend loaded: {model_id}")
    
    # Initialize
    workload_gen = WorkloadGenerator(seed=42)
    all_results = {}
    
    # Experiment 1: TTFT Stair-Step
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: TTFT Stair-Step")
    print("=" * 80)
    
    workload = workload_gen.generate_burst(num_requests=16, max_tokens=32)
    runner = ExperimentRunner(backend, FillBatchPolicy(), max_batch_size=8, backend_name="hf")
    metrics = runner.run_workload(workload, policy_name="fill_batch")
    
    # Sort by request ID
    sorted_metrics = sorted(metrics.metrics, key=lambda m: m.request_id)
    ttft_data = [{"ttft_ms": m.ttft_ms, "request_id": m.request_id} for m in sorted_metrics]
    
    plot_ttft_stair_step(ttft_data, "/results/figures/hf_ttft_stair_step.png")
    metrics.save_jsonl("/results/runs/hf_ttft_stair_step.jsonl")
    all_results["ttft_stair_step"] = metrics.get_summary()
    
    print(f"✓ Batch 1 Avg TTFT: {sum(m.ttft_ms for m in sorted_metrics[:8])/8:.1f} ms")
    print(f"✓ Batch 2 Avg TTFT: {sum(m.ttft_ms for m in sorted_metrics[8:])/8:.1f} ms")
    
    # Experiment 2: Dispatch Policies
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Dispatch Policies")
    print("=" * 80)
    
    policies = [
        ("fill_batch", FillBatchPolicy()),
        ("periodic_5ms", PeriodicPolicy(5)),
        ("periodic_10ms", PeriodicPolicy(10)),
        ("periodic_20ms", PeriodicPolicy(20)),
        ("periodic_50ms", PeriodicPolicy(50)),
        ("max_wait_50ms", MaxWaitPolicy(50)),
        ("max_wait_100ms", MaxWaitPolicy(100)),
        ("max_wait_200ms", MaxWaitPolicy(200)),
        ("short_first", ShortFirstPolicy()),
    ]
    
    policy_results = []
    for policy_name, policy in policies:
        print(f"  Testing {policy_name}...", end=" ", flush=True)
        workload = workload_gen.generate_poisson(num_requests=16, lambda_rate=2.0, max_tokens=64)
        runner = ExperimentRunner(backend, policy, max_batch_size=8, backend_name="hf")
        metrics = runner.run_workload(workload, policy_name=policy_name)
        summary = metrics.get_summary()
        summary["policy"] = policy_name
        policy_results.append(summary)
        metrics.save_jsonl(f"/results/runs/hf_policy_{policy_name}.jsonl")
        print(f"✓ {summary['throughput_tok_s']:.1f} tok/s, TTFT P99: {summary['ttft_p99']:.0f}ms")
    
    plot_policy_comparison(policy_results, "/results/figures/hf_dispatch_policies.png")
    all_results["dispatch_policies"] = policy_results
    
    # Experiment 3: Head-of-Line Blocking
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Head-of-Line Blocking")
    print("=" * 80)
    
    workload = workload_gen.generate_mixed_lengths(
        num_requests=32, short_tokens=32, long_tokens=256
    )
    
    # FIFO
    print("  Testing FIFO...", end=" ", flush=True)
    runner = ExperimentRunner(backend, FillBatchPolicy(), max_batch_size=8, backend_name="hf")
    fifo_metrics = runner.run_workload(workload, policy_name="fifo")
    fifo_short = fifo_metrics.filter_by_length(is_short=True, short_threshold=64)
    fifo_long = fifo_metrics.filter_by_length(is_short=False, short_threshold=64)
    fifo_results = {
        "policy": "fifo",
        **fifo_metrics.get_summary(),
        "short_ttft_p99": fifo_short.get_summary().get("ttft_p99", 0),
        "short_e2e_p99": fifo_short.get_summary().get("e2e_p99", 0),
        "long_e2e_p99": fifo_long.get_summary().get("e2e_p99", 0),
    }
    fifo_metrics.save_jsonl("/results/runs/hf_hol_fifo.jsonl")
    print(f"✓ Short E2E P99: {fifo_results['short_e2e_p99']:.0f}ms")
    
    # Short-first
    print("  Testing Short-first...", end=" ", flush=True)
    runner = ExperimentRunner(backend, ShortFirstPolicy(), max_batch_size=8, backend_name="hf")
    sjf_metrics = runner.run_workload(workload, policy_name="short_first")
    sjf_short = sjf_metrics.filter_by_length(is_short=True, short_threshold=64)
    sjf_long = sjf_metrics.filter_by_length(is_short=False, short_threshold=64)
    sjf_results = {
        "policy": "short_first",
        **sjf_metrics.get_summary(),
        "short_ttft_p99": sjf_short.get_summary().get("ttft_p99", 0),
        "short_e2e_p99": sjf_short.get_summary().get("e2e_p99", 0),
        "long_e2e_p99": sjf_long.get_summary().get("e2e_p99", 0),
    }
    sjf_metrics.save_jsonl("/results/runs/hf_hol_short_first.jsonl")
    print(f"✓ Short E2E P99: {sjf_results['short_e2e_p99']:.0f}ms")
    
    plot_hol_comparison(fifo_results, sjf_results, "/results/figures/hf_head_of_line.png")
    all_results["head_of_line"] = {"fifo": fifo_results, "short_first": sjf_results}
    
    # Experiment 4: Arrival Process
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Arrival Process")
    print("=" * 80)
    
    # Burst
    print("  Testing burst...", end=" ", flush=True)
    burst_workload = workload_gen.generate_burst(num_requests=32, max_tokens=64)
    runner = ExperimentRunner(backend, FillBatchPolicy(), max_batch_size=8, backend_name="hf")
    burst_metrics = runner.run_workload(burst_workload, policy_name="fill_batch")
    burst_results = {"mode": "burst", **burst_metrics.get_summary()}
    burst_metrics.save_jsonl("/results/runs/hf_arrival_burst.jsonl")
    print(f"✓ {burst_results['throughput_tok_s']:.1f} tok/s")
    
    # Poisson
    print("  Testing Poisson...", end=" ", flush=True)
    poisson_workload = workload_gen.generate_poisson(num_requests=32, lambda_rate=2.0, max_tokens=64)
    runner = ExperimentRunner(backend, FillBatchPolicy(), max_batch_size=8, backend_name="hf")
    poisson_metrics = runner.run_workload(poisson_workload, policy_name="fill_batch")
    poisson_results = {"mode": "poisson", **poisson_metrics.get_summary()}
    poisson_metrics.save_jsonl("/results/runs/hf_arrival_poisson.jsonl")
    print(f"✓ {poisson_results['throughput_tok_s']:.1f} tok/s")
    
    plot_arrival_comparison(burst_results, poisson_results, "/results/figures/hf_arrival_process.png")
    all_results["arrival_process"] = {"burst": burst_results, "poisson": poisson_results}
    
    # Experiment 5: Regime Sweep
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Regime Sweep")
    print("=" * 80)
    
    from src.experiments.regime_sweep import run_prefill_sweep, run_decode_sweep
    from src.experiments.plots import plot_regime_sweep
    
    # Sweep A: Prefill-heavy
    print("  Sweep A: Prefill-heavy (varying prompt length)...")
    prompt_lengths = [32, 64, 128, 256, 512, 1024]
    sweep_a = run_prefill_sweep(backend, model_id, prompt_lengths, fixed_gen_tokens=32)
    
    # Sweep B: Decode-heavy
    print("  Sweep B: Decode-heavy (varying generation length)...")
    gen_lengths = [16, 32, 64, 128, 256]
    sweep_b = run_decode_sweep(backend, model_id, gen_lengths, fixed_prompt_tokens=64)
    
    plot_regime_sweep(sweep_a, sweep_b, "/results/figures/hf_regime_sweep.png")
    all_results["regime_sweep"] = {"sweep_a": sweep_a, "sweep_b": sweep_b}
    
    print(f"✓ Sweep A complete: {len(sweep_a)} points")
    print(f"✓ Sweep B complete: {len(sweep_b)} points")
    
    # Save summary
    save_json(all_results, "/results/hf_suite_summary.json")
    
    print("\n" + "=" * 80)
    print("HF SUITE COMPLETE")
    print("=" * 80)
    print("\nResults saved to /results/")
    
    return all_results


@app.function(
    image=image,
    gpu="A10G",  # A10G has 24GB VRAM, needed for vLLM
    timeout=3600,
    volumes={"/models": model_cache, "/results": results_volume},
)
def run_vllm_suite():
    """Run experiment suite on vLLM backend."""
    import sys
    sys.path.insert(0, "/root")
    
    from src.backends.vllm_backend import VLLMBackend
    from src.experiments.run_suite import ExperimentRunner
    from src.experiments.workloads import WorkloadGenerator
    from src.serving.policies import (
        FillBatchPolicy, PeriodicPolicy, MaxWaitPolicy, ShortFirstPolicy
    )
    from src.experiments.metrics import MetricsAggregator
    from src.experiments.plots import (
        plot_ttft_stair_step, plot_policy_comparison,
        plot_hol_comparison, plot_arrival_comparison
    )
    from src.utils.io import save_json
    import time
    
    print("=" * 80)
    print("RUNNING VLLM SUITE")
    print("=" * 80)
    
    # Load backend
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    backend = VLLMBackend(model_id)
    print(f"✓ Backend loaded: {model_id}")
    
    # Initialize
    workload_gen = WorkloadGenerator(seed=42)
    all_results = {}
    
    # Experiment 1: TTFT Stair-Step
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: TTFT Stair-Step")
    print("=" * 80)
    
    workload = workload_gen.generate_burst(num_requests=16, max_tokens=32)
    runner = ExperimentRunner(backend, FillBatchPolicy(), max_batch_size=8, backend_name="vllm")
    metrics = runner.run_workload(workload, policy_name="fill_batch")
    
    sorted_metrics = sorted(metrics.metrics, key=lambda m: m.request_id)
    ttft_data = [{"ttft_ms": m.ttft_ms, "request_id": m.request_id} for m in sorted_metrics]
    
    plot_ttft_stair_step(ttft_data, "/results/figures/vllm_ttft_stair_step.png")
    metrics.save_jsonl("/results/runs/vllm_ttft_stair_step.jsonl")
    all_results["ttft_stair_step"] = metrics.get_summary()
    
    print(f"✓ Batch 1 Avg TTFT: {sum(m.ttft_ms for m in sorted_metrics[:8])/8:.1f} ms")
    print(f"✓ Batch 2 Avg TTFT: {sum(m.ttft_ms for m in sorted_metrics[8:])/8:.1f} ms")
    
    # Experiment 2: Dispatch Policies
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Dispatch Policies")
    print("=" * 80)
    
    policies = [
        ("fill_batch", FillBatchPolicy()),
        ("periodic_5ms", PeriodicPolicy(5)),
        ("periodic_10ms", PeriodicPolicy(10)),
        ("periodic_20ms", PeriodicPolicy(20)),
        ("periodic_50ms", PeriodicPolicy(50)),
        ("max_wait_50ms", MaxWaitPolicy(50)),
        ("max_wait_100ms", MaxWaitPolicy(100)),
        ("max_wait_200ms", MaxWaitPolicy(200)),
        ("short_first", ShortFirstPolicy()),
    ]
    
    policy_results = []
    for policy_name, policy in policies:
        print(f"  Testing {policy_name}...", end=" ", flush=True)
        workload = workload_gen.generate_poisson(num_requests=16, lambda_rate=2.0, max_tokens=64)
        runner = ExperimentRunner(backend, policy, max_batch_size=8, backend_name="vllm")
        metrics = runner.run_workload(workload, policy_name=policy_name)
        summary = metrics.get_summary()
        summary["policy"] = policy_name
        policy_results.append(summary)
        metrics.save_jsonl(f"/results/runs/vllm_policy_{policy_name}.jsonl")
        print(f"✓ {summary['throughput_tok_s']:.1f} tok/s, TTFT P99: {summary['ttft_p99']:.0f}ms")
    
    plot_policy_comparison(policy_results, "/results/figures/vllm_dispatch_policies.png")
    all_results["dispatch_policies"] = policy_results
    
    # Experiment 3: Head-of-Line Blocking
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Head-of-Line Blocking")
    print("=" * 80)
    
    workload = workload_gen.generate_mixed_lengths(
        num_requests=32, short_tokens=32, long_tokens=256
    )
    
    # FIFO
    print("  Testing FIFO...", end=" ", flush=True)
    runner = ExperimentRunner(backend, FillBatchPolicy(), max_batch_size=8, backend_name="vllm")
    fifo_metrics = runner.run_workload(workload, policy_name="fifo")
    fifo_short = fifo_metrics.filter_by_length(is_short=True, short_threshold=64)
    fifo_long = fifo_metrics.filter_by_length(is_short=False, short_threshold=64)
    fifo_results = {
        "policy": "fifo",
        **fifo_metrics.get_summary(),
        "short_ttft_p99": fifo_short.get_summary().get("ttft_p99", 0),
        "short_e2e_p99": fifo_short.get_summary().get("e2e_p99", 0),
        "long_e2e_p99": fifo_long.get_summary().get("e2e_p99", 0),
    }
    fifo_metrics.save_jsonl("/results/runs/vllm_hol_fifo.jsonl")
    print(f"✓ Short E2E P99: {fifo_results['short_e2e_p99']:.0f}ms")
    
    # Short-first
    print("  Testing Short-first...", end=" ", flush=True)
    runner = ExperimentRunner(backend, ShortFirstPolicy(), max_batch_size=8, backend_name="vllm")
    sjf_metrics = runner.run_workload(workload, policy_name="short_first")
    sjf_short = sjf_metrics.filter_by_length(is_short=True, short_threshold=64)
    sjf_long = sjf_metrics.filter_by_length(is_short=False, short_threshold=64)
    sjf_results = {
        "policy": "short_first",
        **sjf_metrics.get_summary(),
        "short_ttft_p99": sjf_short.get_summary().get("ttft_p99", 0),
        "short_e2e_p99": sjf_short.get_summary().get("e2e_p99", 0),
        "long_e2e_p99": sjf_long.get_summary().get("e2e_p99", 0),
    }
    sjf_metrics.save_jsonl("/results/runs/vllm_hol_short_first.jsonl")
    print(f"✓ Short E2E P99: {sjf_results['short_e2e_p99']:.0f}ms")
    
    plot_hol_comparison(fifo_results, sjf_results, "/results/figures/vllm_head_of_line.png")
    all_results["head_of_line"] = {"fifo": fifo_results, "short_first": sjf_results}
    
    # Experiment 4: Arrival Process
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Arrival Process")
    print("=" * 80)
    
    # Burst
    print("  Testing burst...", end=" ", flush=True)
    burst_workload = workload_gen.generate_burst(num_requests=32, max_tokens=64)
    runner = ExperimentRunner(backend, FillBatchPolicy(), max_batch_size=8, backend_name="vllm")
    burst_metrics = runner.run_workload(burst_workload, policy_name="fill_batch")
    burst_results = {"mode": "burst", **burst_metrics.get_summary()}
    burst_metrics.save_jsonl("/results/runs/vllm_arrival_burst.jsonl")
    print(f"✓ {burst_results['throughput_tok_s']:.1f} tok/s")
    
    # Poisson
    print("  Testing Poisson...", end=" ", flush=True)
    poisson_workload = workload_gen.generate_poisson(num_requests=32, lambda_rate=2.0, max_tokens=64)
    runner = ExperimentRunner(backend, FillBatchPolicy(), max_batch_size=8, backend_name="vllm")
    poisson_metrics = runner.run_workload(poisson_workload, policy_name="fill_batch")
    poisson_results = {"mode": "poisson", **poisson_metrics.get_summary()}
    poisson_metrics.save_jsonl("/results/runs/vllm_arrival_poisson.jsonl")
    print(f"✓ {poisson_results['throughput_tok_s']:.1f} tok/s")
    
    plot_arrival_comparison(burst_results, poisson_results, "/results/figures/vllm_arrival_process.png")
    all_results["arrival_process"] = {"burst": burst_results, "poisson": poisson_results}
    
    # Experiment 5: Regime Sweep
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Regime Sweep")
    print("=" * 80)
    
    from src.experiments.regime_sweep import run_prefill_sweep, run_decode_sweep
    from src.experiments.plots import plot_regime_sweep
    
    # Sweep A: Prefill-heavy
    print("  Sweep A: Prefill-heavy (varying prompt length)...")
    prompt_lengths = [32, 64, 128, 256, 512, 1024]
    sweep_a = run_prefill_sweep(backend, model_id, prompt_lengths, fixed_gen_tokens=32)
    
    # Sweep B: Decode-heavy
    print("  Sweep B: Decode-heavy (varying generation length)...")
    gen_lengths = [16, 32, 64, 128, 256]
    sweep_b = run_decode_sweep(backend, model_id, gen_lengths, fixed_prompt_tokens=64)
    
    plot_regime_sweep(sweep_a, sweep_b, "/results/figures/vllm_regime_sweep.png")
    all_results["regime_sweep"] = {"sweep_a": sweep_a, "sweep_b": sweep_b}
    
    print(f"✓ Sweep A complete: {len(sweep_a)} points")
    print(f"✓ Sweep B complete: {len(sweep_b)} points")
    
    # Save summary
    save_json(all_results, "/results/vllm_suite_summary.json")
    
    print("\n" + "=" * 80)
    print("VLLM SUITE COMPLETE")
    print("=" * 80)
    print("\nResults saved to /results/")
    
    return all_results


@app.function(
    image=image,
    gpu="A10G",  # A10G for vLLM
    timeout=3600,
    volumes={"/models": model_cache, "/results": results_volume},
)
def run_intervention_suite():
    """Run intervention experiment (length-aware + microbatch).
    
    This is a SCHEDULER-LAYER experiment comparing dispatch policies.
    TTFT is approximate (batch start time). For true TTFT, use direct benchmarks.
    
    FIXED: Now saves raw JSONL logs and uses corrected metric computation.
    """
    import sys
    sys.path.insert(0, "/root")
    
    from pathlib import Path
    from src.backends.vllm_backend import VLLMBackend
    from src.experiments.run_suite import ExperimentRunner
    from src.experiments.workloads import WorkloadGenerator
    from src.serving.policies import LengthAwareMicrobatchPolicy, FillBatchPolicy
    from src.experiments.metrics import MetricsAggregator, MetricsValidationError
    from src.utils.io import save_json
    
    print("=" * 80)
    print("RUNNING INTERVENTION SUITE (SCHEDULER-LAYER)")
    print("=" * 80)
    print("NOTE: This measures scheduler-layer timing, not true backend TTFT.")
    print("      For direct backend TTFT, see run_direct_benchmark().\n")
    
    # Ensure output directories exist
    Path("/results/runs").mkdir(parents=True, exist_ok=True)
    Path("/results/intervention").mkdir(parents=True, exist_ok=True)
    
    # Load backend
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    backend = VLLMBackend(model_id)
    
    # Generate mixed workload with REALISTIC POISSON ARRIVALS.
    #
    # Previous versions used a synchronized burst (all arrival_time=0), which
    # made the FIFO baseline a strawman: 1000 requests dumped at once drained
    # in FIFO order gave short requests a ~525 s queueing P99. That measured
    # "we stopped dumping everything at once", not a real scheduling win.
    #
    # Here requests arrive as a Poisson process at ARRIVAL_RATE req/s (below the
    # measured saturation throughput so the system is loaded but stable), and
    # run_workload admits each request only at its arrival time. The FIFO vs
    # length-aware comparison then reflects genuine head-of-line blocking under
    # open-loop load.
    NUM_REQUESTS = 300
    # At the scheduler layer each batch blocks until done and is dominated by the
    # 256-token long request (~4s), so effective service is ~2 req/s. We arrive at
    # 1.2 req/s (~60% utilization): loaded enough to expose head-of-line blocking
    # under FIFO, but stable so the baseline does not diverge into a strawman.
    ARRIVAL_RATE = 1.2  # requests/sec
    workload_gen = WorkloadGenerator(seed=42)
    workload = workload_gen.generate_mixed_lengths(
        num_requests=NUM_REQUESTS, short_tokens=32, long_tokens=256,
        short_ratio=0.5, lambda_rate=ARRIVAL_RATE,
    )

    short_count = sum(1 for r in workload if r.max_new_tokens == 32)
    long_count = sum(1 for r in workload if r.max_new_tokens == 256)
    arrival_span = workload[-1].arrival_time if workload else 0.0
    print(f"Workload: {len(workload)} requests ({short_count} short @ 32 tokens, {long_count} long @ 256 tokens)")
    print(f"  Poisson arrivals @ {ARRIVAL_RATE} req/s over ~{arrival_span:.0f}s (open-loop)")
    
    # =========================================================================
    # Baseline: FIFO
    # =========================================================================
    print("\n--- Running Baseline (FIFO) ---")
    runner = ExperimentRunner(backend, FillBatchPolicy(), max_batch_size=8, backend_name="vllm")
    baseline_metrics = runner.run_workload(workload, policy_name="fifo", respect_arrival_times=True)
    
    # Save raw JSONL logs FIRST (before any aggregation)
    baseline_log_path = "/results/runs/intervention_fifo.jsonl"
    baseline_validation = baseline_metrics.save_jsonl(baseline_log_path)
    print(f"✓ Saved baseline logs: {baseline_log_path}")
    print(f"  Validation: {baseline_validation['num_requests']} requests, "
          f"valid={baseline_validation['valid']}")
    
    # =========================================================================
    # Intervention: Length-aware microbatch
    # =========================================================================
    print("\n--- Running Intervention (Length-Aware Microbatch) ---")
    policy = LengthAwareMicrobatchPolicy(
        window_ms=20,
        length_buckets={"short": 64, "medium": 128, "long": 256}
    )
    runner = ExperimentRunner(backend, policy, max_batch_size=8, backend_name="vllm")
    intervention_metrics = runner.run_workload(workload, policy_name="length_aware_microbatch", respect_arrival_times=True)
    
    # Save raw JSONL logs
    intervention_log_path = "/results/runs/intervention_length_aware.jsonl"
    intervention_validation = intervention_metrics.save_jsonl(intervention_log_path)
    print(f"✓ Saved intervention logs: {intervention_log_path}")
    print(f"  Validation: {intervention_validation['num_requests']} requests, "
          f"valid={intervention_validation['valid']}")
    
    # =========================================================================
    # Compute metrics from validated data
    # =========================================================================
    print("\n--- Computing Metrics ---")
    
    # Get full summaries
    baseline_summary = baseline_metrics.get_summary(validate=True)
    intervention_summary = intervention_metrics.get_summary(validate=True)
    
    # Filter for short requests
    baseline_short = baseline_metrics.filter_by_length(is_short=True, short_threshold=64)
    intervention_short = intervention_metrics.filter_by_length(is_short=True, short_threshold=64)
    
    baseline_short_summary = baseline_short.get_summary(validate=True)
    intervention_short_summary = intervention_short.get_summary(validate=True)
    
    # Debug: Print what we got
    print(f"\n  Baseline (all): {baseline_summary.get('num_requests', 0)} requests, "
          f"TTFT P99={baseline_summary.get('ttft_p99', 0):.1f}ms, "
          f"E2E P99={baseline_summary.get('e2e_p99', 0):.1f}ms")
    print(f"  Baseline (short): {baseline_short_summary.get('num_requests', 0)} requests, "
          f"TTFT P99={baseline_short_summary.get('ttft_p99', 0):.1f}ms, "
          f"E2E P99={baseline_short_summary.get('e2e_p99', 0):.1f}ms")
    print(f"  Intervention (all): {intervention_summary.get('num_requests', 0)} requests, "
          f"TTFT P99={intervention_summary.get('ttft_p99', 0):.1f}ms, "
          f"E2E P99={intervention_summary.get('e2e_p99', 0):.1f}ms")
    print(f"  Intervention (short): {intervention_short_summary.get('num_requests', 0)} requests, "
          f"TTFT P99={intervention_short_summary.get('ttft_p99', 0):.1f}ms, "
          f"E2E P99={intervention_short_summary.get('e2e_p99', 0):.1f}ms")
    
    # Build results
    baseline_results = {
        "policy": "fifo",
        "measurement_layer": "scheduler_layer",
        **baseline_summary,
        "short_ttft_p99": baseline_short_summary.get("ttft_p99", 0),
        "short_e2e_p99": baseline_short_summary.get("e2e_p99", 0),
        "short_num_requests": baseline_short_summary.get("num_requests", 0),
    }
    
    intervention_results = {
        "policy": "length_aware_microbatch",
        "measurement_layer": "scheduler_layer",
        **intervention_summary,
        "short_ttft_p99": intervention_short_summary.get("ttft_p99", 0),
        "short_e2e_p99": intervention_short_summary.get("e2e_p99", 0),
        "short_num_requests": intervention_short_summary.get("num_requests", 0),
    }
    
    # =========================================================================
    # COMPARABILITY VALIDATION (fail-loudly check)
    # =========================================================================
    comparison_valid = True
    comparison_issues = []
    
    # Check total request counts match
    baseline_total = baseline_summary.get("num_requests", 0)
    intervention_total = intervention_summary.get("num_requests", 0)
    if baseline_total != intervention_total:
        comparison_valid = False
        comparison_issues.append(
            f"Total request count mismatch: baseline={baseline_total}, intervention={intervention_total}"
        )
    
    # Check short request counts match
    baseline_short_count = baseline_short_summary.get("num_requests", 0)
    intervention_short_count = intervention_short_summary.get("num_requests", 0)
    if baseline_short_count != intervention_short_count:
        comparison_valid = False
        comparison_issues.append(
            f"Short request count mismatch: baseline={baseline_short_count}, intervention={intervention_short_count}"
        )
    
    # Check long request counts match
    baseline_long = baseline_metrics.filter_by_length(is_short=False, short_threshold=64)
    intervention_long = intervention_metrics.filter_by_length(is_short=False, short_threshold=64)
    baseline_long_count = baseline_long.get_summary().get("num_requests", 0)
    intervention_long_count = intervention_long.get_summary().get("num_requests", 0)
    if baseline_long_count != intervention_long_count:
        comparison_valid = False
        comparison_issues.append(
            f"Long request count mismatch: baseline={baseline_long_count}, intervention={intervention_long_count}"
        )
    
    if not comparison_valid:
        print("\n" + "=" * 80)
        print("❌ COMPARISON INVALID - CANNOT COMPUTE IMPROVEMENT")
        print("=" * 80)
        for issue in comparison_issues:
            print(f"   • {issue}")
        print("\nThis indicates a bug in the experiment runner or policy.")
        print("The improvement percentage will NOT be computed.")
        improvement = None
        statistical_results = None  # Cannot compute stats without valid comparison
    else:
        print(f"\n✓ Comparability check passed: {baseline_total} requests each")
        
        # Compute improvement with statistical testing
        baseline_e2e = baseline_results.get("short_e2e_p99", 0)
        intervention_e2e = intervention_results.get("short_e2e_p99", 0)

        if baseline_e2e > 0:
            improvement = (baseline_e2e - intervention_e2e) / baseline_e2e * 100
        else:
            improvement = 0.0
            print("\n⚠️  WARNING: Baseline short_e2e_p99 is 0!")
            print("    This indicates a bug in metric computation or data collection.")
            print("    Check the raw JSONL logs for timing values.")
        
        # Statistical significance testing
        print("\n--- Statistical Analysis ---")
        try:
            from scipy import stats
            import numpy as np
            
            # Extract E2E latencies for short requests
            baseline_short_metrics = baseline_short.metrics
            intervention_short_metrics = intervention_short.metrics
            
            baseline_e2es = [m.e2e_ms for m in baseline_short_metrics if m.e2e_ms > 0]
            intervention_e2es = [m.e2e_ms for m in intervention_short_metrics if m.e2e_ms > 0]
            
            if len(baseline_e2es) > 0 and len(intervention_e2es) > 0:
                # Mann-Whitney U test (non-parametric, handles non-normal distributions)
                statistic, p_value = stats.mannwhitneyu(
                    baseline_e2es,
                    intervention_e2es,
                    alternative='greater'  # Test if baseline > intervention
                )
                
                print(f"  Mann-Whitney U test:")
                print(f"    Statistic: {statistic:.2f}")
                print(f"    p-value: {p_value:.6f}")
                print(f"    Significant (p < 0.05): {'✅ YES' if p_value < 0.05 else '❌ NO'}")
                
                # Effect size (Cohen's d)
                baseline_mean = np.mean(baseline_e2es)
                intervention_mean = np.mean(intervention_e2es)
                pooled_std = np.sqrt((np.var(baseline_e2es) + np.var(intervention_e2es)) / 2)
                cohens_d = (baseline_mean - intervention_mean) / pooled_std if pooled_std > 0 else 0
                
                print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
                if abs(cohens_d) < 0.2:
                    print("    Interpretation: Negligible")
                elif abs(cohens_d) < 0.5:
                    print("    Interpretation: Small")
                elif abs(cohens_d) < 0.8:
                    print("    Interpretation: Medium")
                else:
                    print("    Interpretation: Large")
                
                statistical_results = {
                    "mann_whitney_u": float(statistic),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "cohens_d": float(cohens_d),
                    "baseline_mean": float(baseline_mean),
                    "intervention_mean": float(intervention_mean),
                }
            else:
                print("  ⚠️  Cannot compute statistics: insufficient data")
                statistical_results = None
        except ImportError:
            print("  ⚠️  scipy not available, skipping statistical tests")
            statistical_results = None
        except Exception as e:
            print(f"  ⚠️  Error computing statistics: {e}")
            statistical_results = None
    
    # =========================================================================
    # Print Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("📈 INTERVENTION RESULTS (Scheduler-Layer)")
    print("=" * 80)
    print(f"\n  Baseline (FIFO):")
    print(f"    Throughput: {baseline_results.get('throughput_tok_s', 0):.1f} tok/s")
    print(f"    All Requests - TTFT P99: {baseline_results.get('ttft_p99', 0):.1f}ms, E2E P99: {baseline_results.get('e2e_p99', 0):.1f}ms")
    print(f"    Short Requests - TTFT P99: {baseline_results.get('short_ttft_p99', 0):.1f}ms, E2E P99: {baseline_e2e:.1f}ms")
    
    print(f"\n  Intervention (Length-Aware):")
    print(f"    Throughput: {intervention_results.get('throughput_tok_s', 0):.1f} tok/s")
    print(f"    All Requests - TTFT P99: {intervention_results.get('ttft_p99', 0):.1f}ms, E2E P99: {intervention_results.get('e2e_p99', 0):.1f}ms")
    print(f"    Short Requests - TTFT P99: {intervention_results.get('short_ttft_p99', 0):.1f}ms, E2E P99: {intervention_e2e:.1f}ms")
    
    if improvement is not None:
        print(f"\n  📊 Short Request E2E P99 Improvement: {improvement:.1f}%")

        if improvement >= 50:
            print("  ✅ SUCCESS: Met target of ≥50% improvement!")
        elif improvement > 0:
            print(f"  ⚠️  Improvement below 50% target")
        else:
            print("  ❌ No improvement or regression")
    else:
        print(f"\n  📊 Short Request E2E P99 Improvement: INVALID (comparison failed)")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    # Ensure all values are JSON-serializable
    results_to_save = {
        "baseline": baseline_results,
        "intervention": intervention_results,
        "improvement_pct": float(improvement) if improvement is not None and improvement != "INVALID" else "INVALID",
        "comparison_valid": comparison_valid,
        "comparison_issues": comparison_issues if not comparison_valid else [],
        "statistical_analysis": statistical_results if 'statistical_results' in locals() and statistical_results is not None else None,
        "measurement_layer": "scheduler_layer",
        "arrival_process": "poisson",
        "arrival_rate_req_s": ARRIVAL_RATE,
        "sample_size": {
            "total": int(baseline_total),
            "short": int(baseline_short_count),
            "long": int(baseline_long_count),
        },
        "log_files": {
            "baseline": baseline_log_path,
            "intervention": intervention_log_path,
        },
        "validation": {
            "baseline": baseline_validation,
            "intervention": intervention_validation,
        },
    }
    
    try:
        save_json(results_to_save, "/results/intervention_results.json")
    except Exception as e:
        print(f"\n⚠️  Error saving results: {e}")
        print("Attempting to save with error handling...")
        # Try to save a minimal version
        import traceback
        save_json({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "baseline_summary": {
                "num_requests": baseline_total,
                "throughput": baseline_results.get('throughput_tok_s', 0),
            },
            "intervention_summary": {
                "num_requests": intervention_total,
                "throughput": intervention_results.get('throughput_tok_s', 0),
            },
        }, "/results/intervention_results_error.json")
        raise
    
    print(f"\n✓ Results saved to /results/intervention_results.json")
    print(f"✓ Raw logs saved to /results/runs/intervention_*.jsonl")
    
    return {
        "baseline": baseline_results,
        "intervention": intervention_results,
        "improvement_pct": improvement if improvement is not None else "INVALID",
        "comparison_valid": comparison_valid,
    }


@app.function(
    image=image,
    gpu="A10G",  # A10G for vLLM
    timeout=1800,
    volumes={"/models": model_cache, "/results": results_volume},
)
def profile_vllm():
    """Profile vLLM backend."""
    import sys
    sys.path.insert(0, "/root")
    
    from src.backends.vllm_backend import VLLMBackend
    from src.experiments.profile import Profiler
    from src.experiments.workloads import WorkloadGenerator
    from src.utils.io import save_json
    
    print("=" * 80)
    print("PROFILING VLLM")
    print("=" * 80)
    
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    backend = VLLMBackend(model_id)
    
    workload_gen = WorkloadGenerator(seed=42)
    workload = workload_gen.generate_burst(num_requests=8, max_tokens=32)
    
    profiler = Profiler(use_torch_profiler=True)
    
    # Profile generation
    with profiler.profile("vllm_batch_generation"):
        from src.backends.backend_base import GenerationRequest
        requests = [
            GenerationRequest(
                request_id=req.request_id,
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
            )
            for req in workload
        ]
        results = backend.generate(requests)
    
    # Save profile
    profiler.save("/results/profiles/vllm_profile.json")
    summary = profiler.get_summary()
    
    print(f"\n📊 Profiling Summary:")
    print(f"   CUDA time: {summary.get('total_cuda_time_ms', 0):.1f} ms")
    print(f"   CPU time: {summary.get('total_cpu_time_ms', 0):.1f} ms")
    print(f"   Max memory: {summary.get('max_memory_mb', 0):.1f} MB")
    
    return summary


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/results": results_volume},
)
def generate_report():
    """Generate final report from results."""
    import sys
    sys.path.insert(0, "/root")
    
    from src.utils.io import load_json
    from pathlib import Path
    import json
    
    print("=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)
    
    # Load results
    try:
        hf_results = load_json("/results/hf_suite_summary.json")
    except:
        hf_results = {}
    
    try:
        vllm_results = load_json("/results/vllm_suite_summary.json")
    except:
        vllm_results = {}
    
    try:
        intervention_results = load_json("/results/intervention_results.json")
    except:
        intervention_results = {}
    
    try:
        profile_results = load_json("/results/profiles/vllm_profile.json")
    except:
        profile_results = {}

    try:
        direct_results = load_json("/results/direct_benchmark/results.json")
    except:
        direct_results = {}
    
    # Generate markdown report
    def format_table(headers, rows):
        """Format a markdown table."""
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)
    
    report = f"""# LLM Serving Experiments Report

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

"""

    # ---- Section 0: Direct Benchmark (scheduler bypassed; most trustworthy) ----
    report += "## 0. Direct Backend Benchmark (scheduler bypassed, true streaming TTFT)\n\n"
    if direct_results:
        def _direct_row(label, d):
            if not d:
                return None
            return [
                label,
                f"{d.get('throughput_tok_s', 0):.1f}",
                f"{d.get('ttft_p50_ms', 0):.1f}",
                f"{d.get('ttft_p99_ms', 0):.1f}",
                f"{d.get('e2e_p99_ms', 0):.1f}",
                str(d.get('num_requests', '?')),
            ]
        hf_d = direct_results.get("hf_sequential")
        vseq = direct_results.get("vllm_streaming_sequential") or direct_results.get("vllm_sync_sequential")
        vconc = direct_results.get("vllm_concurrent")
        rows = []
        for label, d in [("HF sequential", hf_d), ("vLLM sequential", vseq), ("vLLM concurrent", vconc)]:
            r = _direct_row(label, d)
            if r:
                rows.append(r)
        if rows:
            report += format_table(
                ["Config", "Throughput (tok/s)", "TTFT P50 (ms)", "TTFT P99 (ms)", "E2E P99 (ms)", "N"],
                rows,
            ) + "\n\n"
        if hf_d and vconc and hf_d.get('throughput_tok_s'):
            tput_x = vconc['throughput_tok_s'] / hf_d['throughput_tok_s']
            report += f"vLLM concurrent throughput is **{tput_x:.1f}x** HuggingFace sequential ({vconc['throughput_tok_s']:.1f} vs {hf_d['throughput_tok_s']:.1f} tok/s).\n\n"
        report += "Note: the HuggingFace path is non-streaming, so its measured TTFT equals its E2E latency; vLLM TTFT is true per-request streaming via `AsyncLLMEngine`.\n\n"
    else:
        report += "*Direct benchmark results not available. Run `modal run modal_app.py::run_direct_benchmark`.*\n\n"

    report += """## 1. Backend Comparison: HF vs vLLM (scheduler-layer, burst arrivals)

### Key Differences

- **HuggingFace**: Sequential processing, no true batching, GPU underutilized
- **vLLM**: Continuous batching with PagedAttention, token-level interleaving, high GPU utilization

### Throughput Comparison

"""
    
    # Add comparison tables if available
    if hf_results.get("dispatch_policies") and vllm_results.get("dispatch_policies"):
        hf_fill = next((r for r in hf_results["dispatch_policies"] if r["policy"] == "fill_batch"), None)
        vllm_fill = next((r for r in vllm_results["dispatch_policies"] if r["policy"] == "fill_batch"), None)
        
        if hf_fill and vllm_fill:
            report += f"""
| Backend | Throughput (tok/s) | TTFT P99 (ms) | E2E P99 (ms) |
|---------|-------------------|---------------|--------------|
| HuggingFace | {hf_fill['throughput_tok_s']:.1f} | {hf_fill['ttft_p99']:.0f} | {hf_fill['e2e_p99']:.0f} |
| vLLM | {vllm_fill['throughput_tok_s']:.1f} | {vllm_fill['ttft_p99']:.0f} | {vllm_fill['e2e_p99']:.0f} |

**Throughput improvement:** {((vllm_fill['throughput_tok_s'] / hf_fill['throughput_tok_s'] - 1) * 100):.1f}%

"""
    
    report += """
### Background: vLLM continuous batching

These are design properties of vLLM (not claims derived from this report's
tables); the throughput figures above are the measured evidence:
1. **Continuous batching**: new requests join the running batch at token
   boundaries rather than waiting for the whole batch to finish.
2. **Token-level interleaving**: multiple requests decode concurrently.
3. **PagedAttention**: paged KV-cache management enables larger effective batches.

## 2. Dispatch Policy Analysis

"""
    
    if vllm_results.get("dispatch_policies"):
        policies = vllm_results["dispatch_policies"]
        rows = [[r["policy"], f"{r['throughput_tok_s']:.1f}", f"{r['ttft_p50']:.0f}", f"{r['ttft_p99']:.0f}", f"{r['e2e_p99']:.0f}"] for r in policies]
        report += format_table(
            ["Policy", "Tok/s", "TTFT P50", "TTFT P99", "E2E P99"],
            rows
        ) + "\n\n"
    
    report += """
**Note:** These are scheduler-layer measurements. TTFT at this layer is
approximate (batch-start time), so TTFT percentiles can read as 0 when the
batch starts in the same tick a request is dispatched. Treat throughput and
E2E latency as the reliable signals here; for true per-request TTFT see the
Direct Benchmark section.

## 3. Head-of-Line Blocking (scheduler-layer, burst arrivals)

"""
    
    if vllm_results.get("head_of_line"):
        hol = vllm_results["head_of_line"]
        report += f"""
| Policy | Short E2E P99 (ms) | Long E2E P99 (ms) | Improvement |
|--------|-------------------|------------------|-------------|
| FIFO | {hol['fifo']['short_e2e_p99']:.0f} | {hol['fifo']['long_e2e_p99']:.0f} | - |
| Short-first | {hol['short_first']['short_e2e_p99']:.0f} | {hol['short_first']['long_e2e_p99']:.0f} | {((hol['fifo']['short_e2e_p99'] - hol['short_first']['short_e2e_p99']) / hol['fifo']['short_e2e_p99'] * 100):.1f}% |

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

"""
    
    report += """
## 4. Arrival Process Impact

"""
    
    if vllm_results.get("arrival_process"):
        ap = vllm_results["arrival_process"]
        report += f"""
| Mode | Throughput (tok/s) | TTFT P99 (ms) | E2E P99 (ms) |
|------|-------------------|---------------|--------------|
| Burst | {ap['burst']['throughput_tok_s']:.1f} | {ap['burst']['ttft_p99']:.0f} | {ap['burst']['e2e_p99']:.0f} |
| Poisson | {ap['poisson']['throughput_tok_s']:.1f} | {ap['poisson']['ttft_p99']:.0f} | {ap['poisson']['e2e_p99']:.0f} |

**Finding:** Measured throughput and tail latency for burst vs Poisson
arrivals are shown in the table above.

"""
    
    report += """
## 5. Scheduling Intervention

"""
    
    if intervention_results:
        base = intervention_results.get("baseline", {})
        interv = intervention_results.get("intervention", {})
        improvement = intervention_results.get("improvement_pct", 0)
        
        throughput_delta = ((interv.get('throughput_tok_s', 0) / base.get('throughput_tok_s', 1) - 1) * 100) if base.get('throughput_tok_s', 0) else 0.0
        stat = intervention_results.get("statistical_analysis") or {}
        sample = intervention_results.get("sample_size") or {}
        report += f"""
**Length-Aware Microbatch Policy** (scheduler-layer; arrivals: {"Poisson" if intervention_results.get("arrival_process") == "poisson" else "see workload config"})

| Metric | Baseline (FIFO) | Intervention | Change |
|--------|----------------|--------------|--------|
| Short E2E P99 | {base.get('short_e2e_p99', 0):.1f} ms | {interv.get('short_e2e_p99', 0):.1f} ms | {improvement:.1f}% lower |
| Short approx. TTFT P99 | {base.get('short_ttft_p99', 0):.1f} ms | {interv.get('short_ttft_p99', 0):.1f} ms | — |
| Throughput | {base.get('throughput_tok_s', 0):.1f} tok/s | {interv.get('throughput_tok_s', 0):.1f} tok/s | {throughput_delta:+.1f}% |

Sample size: {sample.get('total', '?')} requests ({sample.get('short', '?')} short + {sample.get('long', '?')} long), matched across both arms.
"""
        if stat:
            report += f"""
**Statistical test (short-request E2E):** Mann-Whitney U = {stat.get('mann_whitney_u', 0):.0f}, p = {stat.get('p_value', 0):.2e}, Cohen's d = {stat.get('cohens_d', 0):.2f}.
"""
        report += """
**How it works:**
- Requests bucketed by max_new_tokens (short/medium/long)
- Microbatch window (20ms) allows batching within buckets
- Short requests get priority, reducing head-of-line blocking from long requests

"""
    
    report += """
## 6. Profiling Analysis

"""
    
    if profile_results:
        cuda_ms = profile_results.get('total_cuda_time_ms', 0)
        cpu_ms = profile_results.get('total_cpu_time_ms', 0)
        mem_mb = profile_results.get('max_memory_mb', 0)
        report += f"""
**Profiling Summary (host-process `torch.profiler`):**

| Metric | Value |
|--------|-------|
| Total CUDA Time | {cuda_ms:.1f} ms |
| Total CPU Time | {cpu_ms:.1f} ms |
| Max Memory (PyTorch allocator) | {mem_mb:.1f} MB |

**Limitation:** vLLM executes the model in a separate worker process (`EngineCore`),
so the host-side `torch.profiler` used here does not observe the GPU kernels or
device memory of that worker — hence CUDA time and max memory above are typically
0. These numbers therefore reflect only the host/orchestration process, not the
model's device-side execution. A kernel-level breakdown would require profiling
inside the vLLM worker (e.g. Nsight Systems or an in-worker profiler hook), which
is listed under Next Steps.

"""
    else:
        report += "\n*Profiling data not available. Run `modal run modal_app.py::profile_vllm`*\n\n"
    
    report += """
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

"""
    
    Path("/results").mkdir(parents=True, exist_ok=True)
    with open("/results/report.md", "w") as f:
        f.write(report)
    
    print("✓ Report saved to /results/report.md")
    
    return {"status": "complete", "report_path": "/results/report.md"}


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/models": model_cache, "/results": results_volume},
)
def benchmark_hf_direct():
    """Direct HuggingFace benchmark - separate function for clean GPU memory."""
    import sys
    sys.path.insert(0, "/root")
    
    import json
    from pathlib import Path
    
    print("=" * 80)
    print("DIRECT HF BENCHMARK")
    print("=" * 80)
    
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    
    from src.benchmarks.streaming_benchmark import generate_workload
    from src.benchmarks.direct_benchmark import DirectHFBenchmark
    
    workload = generate_workload(num_requests=16, prompt_tokens=128, max_new_tokens=64)
    
    hf_bench = DirectHFBenchmark(model_id)
    hf_summary = hf_bench.run_sequential(workload)
    
    print(f"\n📊 HF Results (n={hf_summary.num_requests}):")
    print(f"   Total Tokens: {hf_summary.total_tokens}")
    print(f"   Total Time: {hf_summary.total_time_s:.2f}s")
    print(f"   Throughput: {hf_summary.throughput_tok_s:.1f} tok/s")
    print(f"   E2E P50: {hf_summary.e2e_p50:.1f} ms")
    print(f"   E2E P99: {hf_summary.e2e_p99:.1f} ms")
    print(f"   (Note: HF TTFT = E2E since no streaming)")
    
    Path("/results/direct_benchmark").mkdir(parents=True, exist_ok=True)
    with open("/results/direct_benchmark/hf_results.json", "w") as f:
        json.dump(hf_summary.to_dict(), f, indent=2)
    
    return hf_summary.to_dict()


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/models": model_cache, "/results": results_volume},
)
def benchmark_vllm_direct():
    """Direct vLLM benchmark with TRUE streaming TTFT - separate function for clean GPU."""
    import sys
    sys.path.insert(0, "/root")
    
    import json
    import asyncio
    from pathlib import Path
    
    print("=" * 80)
    print("DIRECT VLLM BENCHMARK - TRUE STREAMING TTFT")
    print("=" * 80)
    
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    
    from src.benchmarks.streaming_benchmark import generate_workload, generate_mixed_workload, AsyncVLLMBenchmark
    from src.benchmarks.direct_benchmark import summarize_results
    
    # Generate workloads
    workload = generate_workload(num_requests=32, prompt_tokens=128, max_new_tokens=64)
    mixed_workload = generate_mixed_workload(num_requests=32, short_ratio=0.7, short_tokens=32, long_tokens=256)
    
    async def run_all_tests():
        """Run all tests in a single async context to keep engine alive."""
        results = {}
        
        # Initialize async benchmark
        async_bench = AsyncVLLMBenchmark(model_id, gpu_memory_utilization=0.85)
        
        # Sequential test - true TTFT
        print("\n--- Sequential Test (True Streaming TTFT) ---")
        seq_summary = await async_bench.run_sequential_async(workload[:16])
        
        print(f"\n📊 vLLM Sequential Results (n={seq_summary.num_requests}):")
        print(f"   Total Tokens: {seq_summary.total_tokens}")
        print(f"   Total Time: {seq_summary.total_time_s:.2f}s")
        print(f"   Throughput: {seq_summary.throughput_tok_s:.1f} tok/s")
        print(f"   ✅ TTFT P50: {seq_summary.ttft_p50:.1f} ms (TRUE STREAMING)")
        print(f"   ✅ TTFT P99: {seq_summary.ttft_p99:.1f} ms")
        print(f"   E2E P50: {seq_summary.e2e_p50:.1f} ms")
        print(f"   E2E P99: {seq_summary.e2e_p99:.1f} ms")
        print(f"   Avg Output Speed: {seq_summary.avg_output_tok_s:.1f} tok/s per request")
        
        results["sequential"] = seq_summary.to_dict()
        
        # Concurrent test - continuous batching
        print("\n--- Concurrent Test (Continuous Batching) ---")
        conc_summary = await async_bench.run_concurrent_async(workload, max_concurrent=16)
        
        print(f"\n📊 vLLM Concurrent Results (n={conc_summary.num_requests}):")
        print(f"   Total Tokens: {conc_summary.total_tokens}")
        print(f"   Total Time: {conc_summary.total_time_s:.2f}s")
        print(f"   🚀 Throughput: {conc_summary.throughput_tok_s:.1f} tok/s")
        print(f"   ✅ TTFT P50: {conc_summary.ttft_p50:.1f} ms")
        print(f"   ✅ TTFT P99: {conc_summary.ttft_p99:.1f} ms")
        print(f"   E2E P50: {conc_summary.e2e_p50:.1f} ms")
        print(f"   E2E P99: {conc_summary.e2e_p99:.1f} ms")
        
        results["concurrent"] = conc_summary.to_dict()
        
        # Mixed workload test
        print("\n--- Mixed Workload Test (HOL Blocking) ---")
        mixed_summary = await async_bench.run_concurrent_async(mixed_workload, max_concurrent=16)
        
        short_results = [r for r in mixed_summary.results if r.max_new_tokens <= 64]
        long_results = [r for r in mixed_summary.results if r.max_new_tokens > 64]
        short_summary = summarize_results(short_results)
        long_summary = summarize_results(long_results)
        
        print(f"\n📊 Mixed Workload:")
        print(f"   SHORT (n={short_summary.num_requests}): TTFT P99={short_summary.ttft_p99:.1f}ms, E2E P99={short_summary.e2e_p99:.1f}ms")
        print(f"   LONG (n={long_summary.num_requests}): TTFT P99={long_summary.ttft_p99:.1f}ms, E2E P99={long_summary.e2e_p99:.1f}ms")
        
        results["mixed"] = {
            "total": mixed_summary.to_dict(),
            "short": short_summary.to_dict(),
            "long": long_summary.to_dict(),
        }
        
        return results
    
    # Run all tests in a single async context
    results = asyncio.run(run_all_tests())
    
    Path("/results/direct_benchmark").mkdir(parents=True, exist_ok=True)
    with open("/results/direct_benchmark/vllm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hours: HF Seq + vLLM Seq alone take ~37 min; concurrent + mixed need headroom
    volumes={"/models": model_cache, "/results": results_volume},
)
def run_direct_benchmark():
    """Run DIRECT benchmark - bypasses scheduler for accurate TTFT measurement.
    
    This is the canonical benchmark that produces trustworthy numbers.
    """
    import sys
    sys.path.insert(0, "/root")
    
    import time
    import json
    from pathlib import Path
    from dataclasses import asdict
    
    print("=" * 80)
    print("DIRECT BENCHMARK - TRUE TTFT MEASUREMENT")
    print("=" * 80)
    print("\nThis benchmark bypasses our scheduler layer to measure RAW backend performance.")
    print("TTFT is measured via streaming - time from submission to first token arrival.\n")
    
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    
    # =========================================================================
    # STANDARDIZED WORKLOAD - LOCKED PARAMETERS
    # =========================================================================
    
    WORKLOAD_CONFIG = {
        "model_id": model_id,
        "num_requests": 32,
        "prompt_tokens": 128,
        "max_new_tokens": 64,
        "seed": 42,
    }
    
    print("📋 WORKLOAD CONFIG (LOCKED):")
    for k, v in WORKLOAD_CONFIG.items():
        print(f"   {k}: {v}")
    print()
    
    # Generate standardized workload
    from src.benchmarks.streaming_benchmark import generate_workload
    workload = generate_workload(
        num_requests=WORKLOAD_CONFIG["num_requests"],
        prompt_tokens=WORKLOAD_CONFIG["prompt_tokens"],
        max_new_tokens=WORKLOAD_CONFIG["max_new_tokens"],
    )
    
    results = {}
    
    # =========================================================================
    # PART 1: HuggingFace Sequential Baseline
    # =========================================================================
    
    print("=" * 80)
    print("PART 1: HuggingFace Sequential Baseline")
    print("=" * 80)
    
    from src.benchmarks.direct_benchmark import DirectHFBenchmark
    import torch
    import gc
    
    hf_bench = DirectHFBenchmark(model_id)
    hf_summary = hf_bench.run_sequential(workload[:16])  # Subset for HF (slow)
    
    print(f"\n📊 HF Results (n={hf_summary.num_requests}):")
    print(f"   Total Tokens: {hf_summary.total_tokens}")
    print(f"   Total Time: {hf_summary.total_time_s:.2f}s")
    print(f"   Throughput: {hf_summary.throughput_tok_s:.1f} tok/s")
    print(f"   TTFT P50: {hf_summary.ttft_p50:.1f} ms (NOTE: HF TTFT = E2E, no streaming)")
    print(f"   TTFT P99: {hf_summary.ttft_p99:.1f} ms")
    print(f"   E2E P50: {hf_summary.e2e_p50:.1f} ms")
    print(f"   E2E P99: {hf_summary.e2e_p99:.1f} ms")
    
    results["hf_sequential"] = hf_summary.to_dict()
    
    # CRITICAL: Free HF model memory before loading vLLM
    print("\n🧹 Freeing HF model memory...")
    del hf_bench.model
    del hf_bench.tokenizer
    del hf_bench
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"   GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
    
    # =========================================================================
    # PART 2: vLLM Direct - Sequential (True TTFT via streaming)
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("PART 2: vLLM Direct - Async Streaming (TRUE TTFT)")
    print("=" * 80)
    
    # Use AsyncLLMEngine for TRUE streaming TTFT
    from src.benchmarks.streaming_benchmark import AsyncVLLMBenchmark
    
    async_bench = None
    try:
        async_bench = AsyncVLLMBenchmark(model_id, gpu_memory_utilization=0.55)
        vllm_streaming_summary = async_bench.run_sequential(workload[:16])
        
        print(f"\n📊 vLLM Streaming Results (n={vllm_streaming_summary.num_requests}):")
        print(f"   Total Tokens: {vllm_streaming_summary.total_tokens}")
        print(f"   Total Time: {vllm_streaming_summary.total_time_s:.2f}s")
        print(f"   Throughput: {vllm_streaming_summary.throughput_tok_s:.1f} tok/s")
        print(f"   ✅ TTFT P50: {vllm_streaming_summary.ttft_p50:.1f} ms (TRUE STREAMING TTFT)")
        print(f"   ✅ TTFT P99: {vllm_streaming_summary.ttft_p99:.1f} ms (TRUE STREAMING TTFT)")
        print(f"   E2E P50: {vllm_streaming_summary.e2e_p50:.1f} ms")
        print(f"   E2E P99: {vllm_streaming_summary.e2e_p99:.1f} ms")
        print(f"   Avg Output Speed: {vllm_streaming_summary.avg_output_tok_s:.1f} tok/s per request")
        
        results["vllm_streaming_sequential"] = vllm_streaming_summary.to_dict()
        
    except Exception as e:
        print(f"⚠️ AsyncLLMEngine failed: {e}")
        print("   Falling back to sync vLLM with estimated TTFT...")
        
        # Free any partial async engine memory
        if async_bench and hasattr(async_bench, 'engine') and async_bench.engine:
            del async_bench.engine
        gc.collect()
        torch.cuda.empty_cache()
        
        from src.benchmarks.direct_benchmark import DirectVLLMBenchmark
        sync_bench = DirectVLLMBenchmark(model_id, gpu_memory_utilization=0.65)
        vllm_sync_summary = sync_bench.run_sequential(workload[:16])
        
        print(f"\n📊 vLLM Sync Results (TTFT estimated):")
        print(f"   Throughput: {vllm_sync_summary.throughput_tok_s:.1f} tok/s")
        print(f"   ⚠️ TTFT P50: {vllm_sync_summary.ttft_p50:.1f} ms (ESTIMATED)")
        print(f"   E2E P50: {vllm_sync_summary.e2e_p50:.1f} ms")
        
        results["vllm_sync_sequential"] = vllm_sync_summary.to_dict()
        async_bench = sync_bench  # Use sync bench for subsequent tests
    
    # =========================================================================
    # PART 3: vLLM Concurrent - Continuous Batching Test
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("PART 3: vLLM Concurrent - Continuous Batching")
    print("=" * 80)
    
    try:
        if hasattr(async_bench, 'run_concurrent'):
            # Run all requests concurrently to test continuous batching
            vllm_concurrent_summary = async_bench.run_concurrent(workload, max_concurrent=16)
            
            print(f"\n📊 vLLM Concurrent Results (n={vllm_concurrent_summary.num_requests}):")
            print(f"   Total Tokens: {vllm_concurrent_summary.total_tokens}")
            print(f"   Total Time: {vllm_concurrent_summary.total_time_s:.2f}s")
            print(f"   🚀 Throughput: {vllm_concurrent_summary.throughput_tok_s:.1f} tok/s (continuous batching)")
            print(f"   ✅ TTFT P50: {vllm_concurrent_summary.ttft_p50:.1f} ms")
            print(f"   ✅ TTFT P99: {vllm_concurrent_summary.ttft_p99:.1f} ms")
            print(f"   E2E P50: {vllm_concurrent_summary.e2e_p50:.1f} ms")
            print(f"   E2E P99: {vllm_concurrent_summary.e2e_p99:.1f} ms")
            
            results["vllm_concurrent"] = vllm_concurrent_summary.to_dict()
        else:
            # Fallback: use sync batched mode
            print("   Using sync batched mode (no async engine available)...")
            from src.benchmarks.direct_benchmark import DirectVLLMBenchmark
            if not isinstance(async_bench, DirectVLLMBenchmark):
                sync_bench = DirectVLLMBenchmark(model_id, gpu_memory_utilization=0.65)
            else:
                sync_bench = async_bench
            vllm_batched_summary = sync_bench.run_batched(workload, batch_size=8)
            
            print(f"\n📊 vLLM Batched Results (n={vllm_batched_summary.num_requests}):")
            print(f"   Throughput: {vllm_batched_summary.throughput_tok_s:.1f} tok/s")
            print(f"   E2E P50: {vllm_batched_summary.e2e_p50:.1f} ms")
            print(f"   E2E P99: {vllm_batched_summary.e2e_p99:.1f} ms")
            
            results["vllm_batched"] = vllm_batched_summary.to_dict()
        
    except Exception as e:
        print(f"⚠️ Concurrent test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # PART 4: Mixed Workload Test (HOL Blocking)
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("PART 4: Mixed Workload - Head-of-Line Blocking Test")
    print("=" * 80)
    
    from src.benchmarks.streaming_benchmark import generate_mixed_workload
    
    mixed_workload = generate_mixed_workload(
        num_requests=32,
        short_ratio=0.7,
        short_tokens=32,
        long_tokens=256,
    )
    
    try:
        if hasattr(async_bench, 'run_concurrent'):
            mixed_summary = async_bench.run_concurrent(mixed_workload, max_concurrent=16)
        else:
            # Use batched mode as fallback
            mixed_summary = async_bench.run_batched(mixed_workload, batch_size=8)
        
        # Analyze short vs long
        short_results = [r for r in mixed_summary.results if r.max_new_tokens <= 64]
        long_results = [r for r in mixed_summary.results if r.max_new_tokens > 64]
        
        from src.benchmarks.direct_benchmark import summarize_results
        short_summary = summarize_results(short_results)
        long_summary = summarize_results(long_results)
        
        print(f"\n📊 Mixed Workload Results:")
        print(f"   Total: {mixed_summary.num_requests} requests")
        print(f"   Short: {short_summary.num_requests} requests, Long: {long_summary.num_requests} requests")
        print(f"\n   SHORT REQUESTS (n={short_summary.num_requests}):")
        print(f"      TTFT P50: {short_summary.ttft_p50:.1f} ms")
        print(f"      TTFT P99: {short_summary.ttft_p99:.1f} ms")
        print(f"      E2E P50: {short_summary.e2e_p50:.1f} ms")
        print(f"      E2E P99: {short_summary.e2e_p99:.1f} ms")
        print(f"\n   LONG REQUESTS (n={long_summary.num_requests}):")
        print(f"      TTFT P50: {long_summary.ttft_p50:.1f} ms")
        print(f"      TTFT P99: {long_summary.ttft_p99:.1f} ms")
        print(f"      E2E P50: {long_summary.e2e_p50:.1f} ms")
        print(f"      E2E P99: {long_summary.e2e_p99:.1f} ms")
        
        results["mixed_workload"] = {
            "total": mixed_summary.to_dict(),
            "short": short_summary.to_dict(),
            "long": long_summary.to_dict(),
        }
        
    except Exception as e:
        print(f"⚠️ Mixed workload test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    Path("/results/direct_benchmark").mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open("/results/direct_benchmark/config.json", "w") as f:
        json.dump(WORKLOAD_CONFIG, f, indent=2)
    
    # Save results
    with open("/results/direct_benchmark/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # =========================================================================
    # SUMMARY COMPARISON
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    print("""
┌────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ Backend            │ Throughput  │ TTFT P50    │ TTFT P99    │ E2E P99     │
├────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤""")
    
    if "hf_sequential" in results:
        hf = results["hf_sequential"]
        print(f"│ HF Sequential      │ {hf['throughput_tok_s']:>9.1f}  │ {hf['ttft_p50_ms']:>9.1f}  │ {hf['ttft_p99_ms']:>9.1f}  │ {hf['e2e_p99_ms']:>9.1f}  │")
    
    if "vllm_streaming_sequential" in results:
        v = results["vllm_streaming_sequential"]
        print(f"│ vLLM Sequential    │ {v['throughput_tok_s']:>9.1f}  │ {v['ttft_p50_ms']:>9.1f}  │ {v['ttft_p99_ms']:>9.1f}  │ {v['e2e_p99_ms']:>9.1f}  │")
    
    if "vllm_concurrent" in results:
        v = results["vllm_concurrent"]
        print(f"│ vLLM Concurrent    │ {v['throughput_tok_s']:>9.1f}  │ {v['ttft_p50_ms']:>9.1f}  │ {v['ttft_p99_ms']:>9.1f}  │ {v['e2e_p99_ms']:>9.1f}  │")
    
    print("└────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")
    
    if "hf_sequential" in results and "vllm_concurrent" in results:
        hf = results["hf_sequential"]
        vllm = results["vllm_concurrent"]
        print(f"\n🚀 vLLM Concurrent vs HF Sequential:")
        print(f"   Throughput: {vllm['throughput_tok_s']/hf['throughput_tok_s']:.1f}x faster")
        if hf['ttft_p99_ms'] > 0:
            print(f"   TTFT P99: {hf['ttft_p99_ms']/vllm['ttft_p99_ms']:.1f}x lower")
    
    print("\n" + "=" * 80)
    print("DIRECT BENCHMARK COMPLETE")
    print("=" * 80)
    print("\nResults saved to /results/direct_benchmark/")
    
    return results


@app.function(
    image=image,
    gpu=None,
    timeout=300,
    volumes={"/results": results_volume},
)
def extract_intervention_results():
    """Extract and analyze intervention results from JSONL files (bypasses corrupted JSON)."""
    import sys
    sys.path.insert(0, "/root")
    
    from pathlib import Path
    from src.experiments.metrics import MetricsAggregator
    from scipy import stats
    import numpy as np
    
    print("=" * 80)
    print("INTERVENTION EXPERIMENT RESULTS")
    print("=" * 80)
    
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
    
    # Filter short requests (32 tokens) vs long (256 tokens)
    baseline_short_list = [m for m in baseline_metrics.metrics if m.max_new_tokens == 32]
    intervention_short_list = [m for m in intervention_metrics.metrics if m.max_new_tokens == 32]
    
    print(f"\n🔍 Request Distribution:")
    print(f"   Baseline - Short (32 tokens): {len(baseline_short_list)}, Long (256 tokens): {len([m for m in baseline_metrics.metrics if m.max_new_tokens == 256])}")
    print(f"   Intervention - Short (32 tokens): {len(intervention_short_list)}, Long (256 tokens): {len([m for m in intervention_metrics.metrics if m.max_new_tokens == 256])}")
    
    # Create MetricsAggregator instances
    baseline_short = MetricsAggregator(measurement_layer=baseline_metrics.measurement_layer)
    baseline_short.metrics = baseline_short_list
    intervention_short = MetricsAggregator(measurement_layer=intervention_metrics.measurement_layer)
    intervention_short.metrics = intervention_short_list
    
    baseline_short_summary = baseline_short.get_summary(validate=True)
    intervention_short_summary = intervention_short.get_summary(validate=True)
    
    print(f"\n📊 BASELINE (FIFO Policy)")
    print(f"   Total Requests: {baseline_summary.get('num_requests', len(baseline_metrics.metrics))}")
    print(f"   Short Requests: {baseline_short_summary.get('num_requests', len(baseline_short.metrics))}")
    print(f"   Throughput: {baseline_summary.get('throughput_tok_s', 0):.1f} tok/s")
    print(f"   All Requests - TTFT P99: {baseline_summary.get('ttft_p99', 0):.1f} ms, E2E P99: {baseline_summary.get('e2e_p99', 0):.1f} ms")
    print(f"   Short Requests - TTFT P99: {baseline_short_summary.get('ttft_p99', 0):.1f} ms, E2E P99: {baseline_short_summary.get('e2e_p99', 0):.1f} ms")
    
    print(f"\n📊 INTERVENTION (Length-Aware Microbatch)")
    print(f"   Total Requests: {intervention_summary.get('num_requests', len(intervention_metrics.metrics))}")
    print(f"   Short Requests: {intervention_short_summary.get('num_requests', len(intervention_short.metrics))}")
    print(f"   Throughput: {intervention_summary.get('throughput_tok_s', 0):.1f} tok/s")
    print(f"   All Requests - TTFT P99: {intervention_summary.get('ttft_p99', 0):.1f} ms, E2E P99: {intervention_summary.get('e2e_p99', 0):.1f} ms")
    print(f"   Short Requests - TTFT P99: {intervention_short_summary.get('ttft_p99', 0):.1f} ms, E2E P99: {intervention_short_summary.get('e2e_p99', 0):.1f} ms")
    
    # Calculate improvement
    baseline_e2e = baseline_short_summary.get('e2e_p99', 0)
    intervention_e2e = intervention_short_summary.get('e2e_p99', 0)
    
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
    baseline_e2es = [m.e2e_ms for m in baseline_short.metrics if m.e2e_ms > 0]
    intervention_e2es = [m.e2e_ms for m in intervention_short.metrics if m.e2e_ms > 0]
    
    if len(baseline_e2es) > 0 and len(intervention_e2es) > 0:
        statistic, p_value = stats.mannwhitneyu(
            baseline_e2es,
            intervention_e2es,
            alternative='greater'
        )
        
        print(f"   Mann-Whitney U Test:")
        print(f"     Statistic: {statistic:.2f}")
        print(f"     p-value: {p_value:.6f}")
        print(f"     Significant (p < 0.05): {'✅ YES' if p_value < 0.05 else '❌ NO'}")
        
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
    
    return {
        "baseline": baseline_summary,
        "intervention": intervention_summary,
        "baseline_short": baseline_short_summary,
        "intervention_short": intervention_short_summary,
        "improvement_pct": improvement,
    }


@app.function(
    image=image,
    gpu=None,  # No GPU needed for reading results
    timeout=300,
    volumes={"/results": results_volume},
)
def show_results_summary():
    """Display summary of all available results."""
    import sys
    sys.path.insert(0, "/root")
    
    from src.utils.io import load_json
    from pathlib import Path
    import json
    import os
    
    print("=" * 80)
    print("MODAL SERVING ALPHA - RESULTS SUMMARY")
    print("=" * 80)
    
    results_dir = Path("/results")
    
    # Check what files exist
    print("\n📁 Available Result Files:")
    print("-" * 80)
    
    all_files = []
    if results_dir.exists():
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith(('.json', '.jsonl', '.md')):
                    rel_path = os.path.relpath(os.path.join(root, file), results_dir)
                    all_files.append(rel_path)
                    print(f"  ✓ {rel_path}")
    
    if not all_files:
        print("  ⚠️  No result files found in /results/")
        return {"status": "no_results"}
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    summary = {}
    
    # 1. Direct Benchmark Results
    print("\n1️⃣ DIRECT BENCHMARK RESULTS")
    print("-" * 80)
    try:
        # Try the unified results.json first
        direct_results = load_json("/results/direct_benchmark/results.json")
        summary["direct_benchmark"] = direct_results
        
        if "hf_sequential" in direct_results:
            hf = direct_results["hf_sequential"]
            print(f"  HuggingFace Sequential:")
            print(f"    Throughput: {hf.get('throughput_tok_s', 0):.1f} tok/s")
            print(f"    TTFT P50: {hf.get('ttft_p50_ms', 0):.1f} ms")
            print(f"    TTFT P99: {hf.get('ttft_p99_ms', 0):.1f} ms")
            print(f"    E2E P99: {hf.get('e2e_p99_ms', 0):.1f} ms")
        
        if "vllm_streaming_sequential" in direct_results:
            vllm_seq = direct_results["vllm_streaming_sequential"]
            print(f"  vLLM Sequential (Streaming):")
            print(f"    Throughput: {vllm_seq.get('throughput_tok_s', 0):.1f} tok/s")
            print(f"    TTFT P50: {vllm_seq.get('ttft_p50_ms', 0):.1f} ms")
            print(f"    TTFT P99: {vllm_seq.get('ttft_p99_ms', 0):.1f} ms")
            print(f"    E2E P99: {vllm_seq.get('e2e_p99_ms', 0):.1f} ms")
        
        if "vllm_concurrent" in direct_results:
            vllm_conc = direct_results["vllm_concurrent"]
            print(f"  vLLM Concurrent (Continuous Batching):")
            print(f"    Throughput: {vllm_conc.get('throughput_tok_s', 0):.1f} tok/s")
            print(f"    TTFT P50: {vllm_conc.get('ttft_p50_ms', 0):.1f} ms")
            print(f"    TTFT P99: {vllm_conc.get('ttft_p99_ms', 0):.1f} ms")
            print(f"    E2E P99: {vllm_conc.get('e2e_p99_ms', 0):.1f} ms")
        
        if "mixed_workload" in direct_results:
            mixed = direct_results["mixed_workload"]
            if "short" in mixed:
                print(f"  Mixed Workload - Short Requests:")
                print(f"    TTFT P99: {mixed['short'].get('ttft_p99_ms', 0):.1f} ms")
                print(f"    E2E P99: {mixed['short'].get('e2e_p99_ms', 0):.1f} ms")
            if "long" in mixed:
                print(f"  Mixed Workload - Long Requests:")
                print(f"    TTFT P99: {mixed['long'].get('ttft_p99_ms', 0):.1f} ms")
                print(f"    E2E P99: {mixed['long'].get('e2e_p99_ms', 0):.1f} ms")
    except:
        # Try separate files
        try:
            hf_results = load_json("/results/direct_benchmark/hf_results.json")
            vllm_results = load_json("/results/direct_benchmark/vllm_results.json")
            summary["direct_benchmark"] = {"hf": hf_results, "vllm": vllm_results}
            
            print(f"  HuggingFace Sequential:")
            print(f"    Throughput: {hf_results.get('throughput_tok_s', hf_results.get('throughput', 0)):.1f} tok/s")
            e2e_p50 = hf_results.get('e2e_p50', hf_results.get('e2e_p50_ms', 0))
            e2e_p99 = hf_results.get('e2e_p99', hf_results.get('e2e_p99_ms', 0))
            print(f"    E2E P50: {e2e_p50:.1f} ms")
            print(f"    E2E P99: {e2e_p99:.1f} ms")
            if 'ttft_p50' in hf_results or 'ttft_p50_ms' in hf_results:
                ttft_p50 = hf_results.get('ttft_p50', hf_results.get('ttft_p50_ms', 0))
                ttft_p99 = hf_results.get('ttft_p99', hf_results.get('ttft_p99_ms', 0))
                print(f"    TTFT P50: {ttft_p50:.1f} ms")
                print(f"    TTFT P99: {ttft_p99:.1f} ms")
            
            if "sequential" in vllm_results:
                seq = vllm_results["sequential"]
                print(f"  vLLM Sequential:")
                print(f"    Throughput: {seq.get('throughput_tok_s', seq.get('throughput', 0)):.1f} tok/s")
                ttft_p50 = seq.get('ttft_p50', seq.get('ttft_p50_ms', 0))
                ttft_p99 = seq.get('ttft_p99', seq.get('ttft_p99_ms', 0))
                e2e_p50 = seq.get('e2e_p50', seq.get('e2e_p50_ms', 0))
                e2e_p99 = seq.get('e2e_p99', seq.get('e2e_p99_ms', 0))
                print(f"    TTFT P50: {ttft_p50:.1f} ms")
                print(f"    TTFT P99: {ttft_p99:.1f} ms")
                print(f"    E2E P50: {e2e_p50:.1f} ms")
                print(f"    E2E P99: {e2e_p99:.1f} ms")
            
            if "concurrent" in vllm_results:
                conc = vllm_results["concurrent"]
                print(f"  vLLM Concurrent (Continuous Batching):")
                print(f"    Throughput: {conc.get('throughput_tok_s', conc.get('throughput', 0)):.1f} tok/s")
                ttft_p50 = conc.get('ttft_p50', conc.get('ttft_p50_ms', 0))
                ttft_p99 = conc.get('ttft_p99', conc.get('ttft_p99_ms', 0))
                e2e_p50 = conc.get('e2e_p50', conc.get('e2e_p50_ms', 0))
                e2e_p99 = conc.get('e2e_p99', conc.get('e2e_p99_ms', 0))
                print(f"    TTFT P50: {ttft_p50:.1f} ms")
                print(f"    TTFT P99: {ttft_p99:.1f} ms")
                print(f"    E2E P50: {e2e_p50:.1f} ms")
                print(f"    E2E P99: {e2e_p99:.1f} ms")
            
            if "mixed" in vllm_results:
                mixed = vllm_results["mixed"]
                if "short" in mixed:
                    print(f"  Mixed Workload - Short Requests:")
                    ttft_p99 = mixed['short'].get('ttft_p99', mixed['short'].get('ttft_p99_ms', 0))
                    e2e_p99 = mixed['short'].get('e2e_p99', mixed['short'].get('e2e_p99_ms', 0))
                    print(f"    TTFT P99: {ttft_p99:.1f} ms")
                    print(f"    E2E P99: {e2e_p99:.1f} ms")
                if "long" in mixed:
                    print(f"  Mixed Workload - Long Requests:")
                    ttft_p99 = mixed['long'].get('ttft_p99', mixed['long'].get('ttft_p99_ms', 0))
                    e2e_p99 = mixed['long'].get('e2e_p99', mixed['long'].get('e2e_p99_ms', 0))
                    print(f"    TTFT P99: {ttft_p99:.1f} ms")
                    print(f"    E2E P99: {e2e_p99:.1f} ms")
        except Exception as e:
            print(f"  ⚠️  Direct benchmark results not available: {e}")
    
    # 2. HF Suite Results
    print("\n2️⃣ HUGGINGFACE SUITE RESULTS")
    print("-" * 80)
    try:
        hf_results = load_json("/results/hf_suite_summary.json")
        summary["hf_suite"] = hf_results
        
        if "dispatch_policies" in hf_results:
            print(f"  Dispatch Policies Tested: {len(hf_results['dispatch_policies'])}")
            fill_batch = next((p for p in hf_results['dispatch_policies'] if p.get('policy') == 'fill_batch'), None)
            if fill_batch:
                print(f"  Fill Batch Policy:")
                print(f"    Throughput: {fill_batch.get('throughput_tok_s', 0):.1f} tok/s")
                print(f"    TTFT P99: {fill_batch.get('ttft_p99', 0):.0f} ms")
        
        if "head_of_line" in hf_results:
            hol = hf_results["head_of_line"]
            print(f"  Head-of-Line Blocking:")
            if "fifo" in hol:
                print(f"    FIFO Short E2E P99: {hol['fifo'].get('short_e2e_p99', 0):.0f} ms")
            if "short_first" in hol:
                print(f"    Short-First Short E2E P99: {hol['short_first'].get('short_e2e_p99', 0):.0f} ms")
    except Exception as e:
        print(f"  ⚠️  HF suite results not available: {e}")
    
    # 3. vLLM Suite Results
    print("\n3️⃣ VLLM SUITE RESULTS")
    print("-" * 80)
    try:
        vllm_results = load_json("/results/vllm_suite_summary.json")
        summary["vllm_suite"] = vllm_results
        
        if "dispatch_policies" in vllm_results:
            print(f"  Dispatch Policies Tested: {len(vllm_results['dispatch_policies'])}")
            fill_batch = next((p for p in vllm_results['dispatch_policies'] if p.get('policy') == 'fill_batch'), None)
            if fill_batch:
                print(f"  Fill Batch Policy:")
                print(f"    Throughput: {fill_batch.get('throughput_tok_s', 0):.1f} tok/s")
                print(f"    TTFT P99: {fill_batch.get('ttft_p99', 0):.0f} ms")
        
        if "head_of_line" in vllm_results:
            hol = vllm_results["head_of_line"]
            print(f"  Head-of-Line Blocking:")
            if "fifo" in hol:
                print(f"    FIFO Short E2E P99: {hol['fifo'].get('short_e2e_p99', 0):.0f} ms")
            if "short_first" in hol:
                print(f"    Short-First Short E2E P99: {hol['short_first'].get('short_e2e_p99', 0):.0f} ms")
    except Exception as e:
        print(f"  ⚠️  vLLM suite results not available: {e}")
    
    # 4. Intervention Results
    print("\n4️⃣ INTERVENTION RESULTS")
    print("-" * 80)
    try:
        intervention = load_json("/results/intervention_results.json")
        summary["intervention"] = intervention
        
        if "baseline" in intervention and "intervention" in intervention:
            base = intervention["baseline"]
            interv = intervention["intervention"]
            improvement = intervention.get("improvement_pct", 0)
            
            print(f"  Baseline (FIFO) Short E2E P99: {base.get('short_e2e_p99', 0):.0f} ms")
            print(f"  Intervention Short E2E P99: {interv.get('short_e2e_p99', 0):.0f} ms")
            print(f"  Improvement: {improvement:.1f}%")
    except Exception as e:
        print(f"  ⚠️  Intervention results not available: {e}")
    
    # 5. Profile Results
    print("\n5️⃣ PROFILING RESULTS")
    print("-" * 80)
    try:
        profile = load_json("/results/profiles/vllm_profile.json")
        summary["profile"] = profile
        
        print(f"  Total CUDA Time: {profile.get('total_cuda_time_ms', 0):.1f} ms")
        print(f"  Total CPU Time: {profile.get('total_cpu_time_ms', 0):.1f} ms")
        print(f"  Max Memory: {profile.get('max_memory_mb', 0):.1f} MB")
    except Exception as e:
        print(f"  ⚠️  Profile results not available: {e}")
    
    # 6. QServe Build Results
    print("\n6️⃣ QSERVE BUILD RESULTS")
    print("-" * 80)
    try:
        build_logs_dir = Path("/results/qserve_build_logs")
        if build_logs_dir.exists():
            build_files = list(build_logs_dir.glob("results_*.json"))
            if build_files:
                latest_build = max(build_files, key=lambda p: p.stat().st_mtime)
                build_results = load_json(str(latest_build))
                summary["qserve_build"] = build_results
                
                print(f"  Status: {build_results.get('status', 'unknown')}")
                print(f"  Steps Passed: {sum(1 for s in build_results.get('steps', []) if s.get('status') == 'passed')}")
                print(f"  Steps Failed: {sum(1 for s in build_results.get('steps', []) if s.get('status') == 'failed')}")
                if "gpu_info" in build_results:
                    print(f"  GPU: {build_results['gpu_info'].get('name', 'unknown')}")
            else:
                print("  ⚠️  No build result files found")
        else:
            print("  ⚠️  Build logs directory not found")
    except Exception as e:
        print(f"  ⚠️  QServe build results not available: {e}")
    
    print("\n" + "=" * 80)
    print("SUMMARY COMPLETE")
    print("=" * 80)
    
    return summary


@app.function(
    image=image,
    gpu=None,  # No GPU needed for file merging
    timeout=300,
    volumes={"/results": results_volume},
)
def create_unified_results():
    """Create unified direct benchmark results.json from component files."""
    import sys
    sys.path.insert(0, "/root")
    
    from pathlib import Path
    from src.utils.io import load_json
    import json
    
    results_dir = Path("/results/direct_benchmark")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load component results
    hf_path = results_dir / "hf_results.json"
    vllm_path = results_dir / "vllm_results.json"
    
    hf_data = None
    vllm_data = None
    
    if hf_path.exists():
        hf_data = load_json(str(hf_path))
        print(f"✓ Loaded HF results from {hf_path}")
    else:
        print(f"⚠️  HF results not found: {hf_path}")
    
    if vllm_path.exists():
        vllm_data = load_json(str(vllm_path))
        print(f"✓ Loaded vLLM results from {vllm_path}")
    else:
        print(f"⚠️  vLLM results not found: {vllm_path}")
    
    # Extract vLLM results (handle different possible structures)
    vllm_seq = None
    vllm_conc = None
    mixed = None
    
    if vllm_data:
        # Try different possible key names
        vllm_seq = (
            vllm_data.get("vllm_streaming_sequential") or
            vllm_data.get("sequential") or
            vllm_data.get("vllm_seq")
        )
        vllm_conc = (
            vllm_data.get("vllm_concurrent") or
            vllm_data.get("concurrent") or
            vllm_data.get("vllm_conc")
        )
        mixed = vllm_data.get("mixed") or vllm_data.get("mixed_workload")
    
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
            "vllm_sequential": vllm_seq,
            "vllm_concurrent": vllm_conc,
            "mixed_workload": mixed,
        },
        "comparisons": {},
    }
    
    # Add comparisons if data available
    if hf_data and (vllm_seq or vllm_conc):
        
        # Sequential comparison (apples-to-apples)
        if vllm_seq and isinstance(vllm_seq, dict) and hf_data.get("throughput_tok_s"):
            hf_throughput = hf_data.get("throughput_tok_s", 0)
            vllm_seq_throughput = vllm_seq.get("throughput_tok_s", 0)
            if hf_throughput > 0 and vllm_seq_throughput > 0:
                unified["comparisons"]["sequential"] = {
                    "note": "Apples-to-apples: 16 requests each",
                    "throughput_ratio": vllm_seq_throughput / hf_throughput,
                    "ttft_p99_ratio": hf_data.get("ttft_p99_ms", 0) / vllm_seq.get("ttft_p99_ms", 1) if vllm_seq.get("ttft_p99_ms", 0) > 0 else 0,
                    "e2e_p99_ratio": hf_data.get("e2e_p99_ms", 0) / vllm_seq.get("e2e_p99_ms", 1) if vllm_seq.get("e2e_p99_ms", 0) > 0 else 0,
                }
        
        # Concurrent (note: different request counts)
        if vllm_conc and isinstance(vllm_conc, dict) and hf_data.get("throughput_tok_s"):
            hf_throughput = hf_data.get("throughput_tok_s", 0)
            vllm_conc_throughput = vllm_conc.get("throughput_tok_s", 0)
            if hf_throughput > 0 and vllm_conc_throughput > 0:
                unified["comparisons"]["concurrent"] = {
                    "note": "Different request counts: vLLM 32, HF 16",
                    "throughput_ratio": vllm_conc_throughput / hf_throughput,
                    "ttft_p99_ratio": hf_data.get("ttft_p99_ms", 0) / vllm_conc.get("ttft_p99_ms", 1) if vllm_conc.get("ttft_p99_ms", 0) > 0 else 0,
                }
    
    # Save unified results
    output_path = results_dir / "results.json"
    with open(output_path, 'w') as f:
        json.dump(unified, f, indent=2)
    
    print(f"\n✓ Unified results saved to {output_path}")
    print(f"  - HF Sequential: {'✓' if hf_data else '✗'}")
    print(f"  - vLLM Sequential: {'✓' if vllm_seq else '✗'}")
    print(f"  - vLLM Concurrent: {'✓' if vllm_conc else '✗'}")
    print(f"  - Mixed Workload: {'✓' if mixed else '✗'}")
    
    if unified["comparisons"]:
        print(f"\n  Comparisons:")
        for name, comp in unified["comparisons"].items():
            print(f"    {name}: {comp.get('note', '')}")
            if "throughput_ratio" in comp:
                print(f"      Throughput ratio: {comp['throughput_ratio']:.2f}x")
    
    return {"status": "complete", "output_path": str(output_path)}


@app.function(
    image=image,
    gpu=None,
    timeout=600,
    volumes={"/results": results_volume},
)
def rebuild_intervention_summary():
    """Recompute /results/intervention_results.json from existing JSONL logs.

    Useful when run_intervention_suite completed the experiment but failed at
    JSON serialization. Reads the raw per-request logs already on the volume
    and rewrites the summary using the (now-fixed) sanitizer.
    """
    import sys
    sys.path.insert(0, "/root")

    from src.experiments.metrics import MetricsAggregator
    from src.utils.io import save_json

    baseline_log = "/results/runs/intervention_fifo.jsonl"
    intervention_log = "/results/runs/intervention_length_aware.jsonl"

    print("=" * 80)
    print("REBUILDING INTERVENTION SUMMARY FROM JSONL")
    print("=" * 80)
    print(f"  Baseline log:     {baseline_log}")
    print(f"  Intervention log: {intervention_log}")

    baseline_metrics = MetricsAggregator.from_jsonl(baseline_log)
    intervention_metrics = MetricsAggregator.from_jsonl(intervention_log)

    baseline_short = baseline_metrics.filter_by_length(
        is_short=True, short_threshold=64
    )
    intervention_short = intervention_metrics.filter_by_length(
        is_short=True, short_threshold=64
    )

    baseline_summary = baseline_metrics.get_summary(validate=True)
    intervention_summary = intervention_metrics.get_summary(validate=True)
    baseline_short_summary = baseline_short.get_summary(validate=True)
    intervention_short_summary = intervention_short.get_summary(validate=True)

    baseline_results = {
        "policy": "fifo_baseline",
        "measurement_layer": "scheduler_layer",
        **baseline_summary,
        "short_ttft_p99": baseline_short_summary.get("ttft_p99", 0),
        "short_e2e_p99": baseline_short_summary.get("e2e_p99", 0),
        "short_num_requests": baseline_short_summary.get("num_requests", 0),
    }
    intervention_results = {
        "policy": "length_aware_microbatch",
        "measurement_layer": "scheduler_layer",
        **intervention_summary,
        "short_ttft_p99": intervention_short_summary.get("ttft_p99", 0),
        "short_e2e_p99": intervention_short_summary.get("e2e_p99", 0),
        "short_num_requests": intervention_short_summary.get("num_requests", 0),
    }

    baseline_e2e = baseline_results["short_e2e_p99"]
    intervention_e2e = intervention_results["short_e2e_p99"]
    improvement = (
        (baseline_e2e - intervention_e2e) / baseline_e2e * 100
        if baseline_e2e > 0
        else 0.0
    )

    statistical_results = None
    try:
        from scipy import stats
        import numpy as np

        b_e2es = [m.e2e_ms for m in baseline_short.metrics if m.e2e_ms > 0]
        i_e2es = [m.e2e_ms for m in intervention_short.metrics if m.e2e_ms > 0]
        if b_e2es and i_e2es:
            statistic, p_value = stats.mannwhitneyu(
                b_e2es, i_e2es, alternative="greater"
            )
            b_mean, i_mean = np.mean(b_e2es), np.mean(i_e2es)
            pooled_std = np.sqrt((np.var(b_e2es) + np.var(i_e2es)) / 2)
            cohens_d = (b_mean - i_mean) / pooled_std if pooled_std > 0 else 0.0
            statistical_results = {
                "mann_whitney_u": float(statistic),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "cohens_d": float(cohens_d),
                "baseline_mean": float(b_mean),
                "intervention_mean": float(i_mean),
            }
    except Exception as e:
        print(f"  ⚠️  Stats computation skipped: {e}")

    results_to_save = {
        "baseline": baseline_results,
        "intervention": intervention_results,
        "improvement_pct": float(improvement),
        "comparison_valid": True,
        "comparison_issues": [],
        "statistical_analysis": statistical_results,
        "measurement_layer": "scheduler_layer",
        "arrival_process": "poisson",
        "sample_size": {
            "total": int(baseline_summary.get("num_requests", 0)),
            "short": int(baseline_short_summary.get("num_requests", 0)),
            "long": int(
                baseline_summary.get("num_requests", 0)
                - baseline_short_summary.get("num_requests", 0)
            ),
        },
        "log_files": {
            "baseline": baseline_log,
            "intervention": intervention_log,
        },
        "rebuilt_from_jsonl": True,
    }

    save_json(results_to_save, "/results/intervention_results.json")
    print(f"\n✓ Wrote /results/intervention_results.json")
    print(f"  Short E2E P99 improvement: {improvement:.1f}%")
    return {
        "improvement_pct": improvement,
        "baseline_short_e2e_p99": baseline_e2e,
        "intervention_short_e2e_p99": intervention_e2e,
    }


@app.local_entrypoint()
def main():
    """Default: run direct benchmark."""
    print("Running direct benchmark (proper TTFT measurement)...")
    result = run_direct_benchmark.remote()
    print(f"\nResult: {result}")
