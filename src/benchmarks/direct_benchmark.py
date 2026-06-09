"""Direct benchmark - bypasses scheduler layer for accurate measurements.

This module measures raw backend performance with proper TTFT instrumentation.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, AsyncIterator
from collections import defaultdict
import statistics


@dataclass
class BenchmarkRequest:
    """A single benchmark request with fixed parameters."""
    request_id: str
    prompt: str
    max_new_tokens: int
    prompt_tokens: int = 0  # Filled by benchmark


@dataclass 
class BenchmarkResult:
    """Per-request timing result."""
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    max_new_tokens: int
    
    # Critical timings (all in seconds, using time.perf_counter())
    submit_time: float  # When request was submitted to engine
    first_token_time: float  # When first token was received
    end_time: float  # When generation completed
    
    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds."""
        return (self.first_token_time - self.submit_time) * 1000
    
    @property
    def e2e_ms(self) -> float:
        """End-to-end latency in milliseconds."""
        return (self.end_time - self.submit_time) * 1000
    
    @property
    def generation_time_ms(self) -> float:
        """Time spent generating tokens (after first token)."""
        return (self.end_time - self.first_token_time) * 1000
    
    @property
    def tokens_per_second(self) -> float:
        """Output generation speed."""
        gen_time = self.end_time - self.first_token_time
        if gen_time > 0 and self.generated_tokens > 1:
            return (self.generated_tokens - 1) / gen_time  # Exclude first token
        return 0.0


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results."""
    num_requests: int
    total_tokens: int
    total_time_s: float
    
    # TTFT stats (ms)
    ttft_min: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    ttft_max: float
    
    # E2E stats (ms)
    e2e_min: float
    e2e_p50: float
    e2e_p95: float
    e2e_p99: float
    e2e_max: float
    
    # Throughput
    throughput_tok_s: float
    throughput_req_s: float
    avg_output_tok_s: float
    
    # Raw results for further analysis
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "num_requests": self.num_requests,
            "total_tokens": self.total_tokens,
            "total_time_s": self.total_time_s,
            "ttft_min_ms": self.ttft_min,
            "ttft_p50_ms": self.ttft_p50,
            "ttft_p95_ms": self.ttft_p95,
            "ttft_p99_ms": self.ttft_p99,
            "ttft_max_ms": self.ttft_max,
            "e2e_min_ms": self.e2e_min,
            "e2e_p50_ms": self.e2e_p50,
            "e2e_p95_ms": self.e2e_p95,
            "e2e_p99_ms": self.e2e_p99,
            "e2e_max_ms": self.e2e_max,
            "throughput_tok_s": self.throughput_tok_s,
            "throughput_req_s": self.throughput_req_s,
            "avg_output_tok_s": self.avg_output_tok_s,
        }


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile using proper quantile method with linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    # Proper quantile: (n-1) * p/100 for 0-indexed array
    position = (n - 1) * p / 100
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, n - 1)
    weight = position - lower_idx
    
    if lower_idx == upper_idx:
        return sorted_data[lower_idx]
    # Linear interpolation
    return sorted_data[lower_idx] * (1 - weight) + sorted_data[upper_idx] * weight


def summarize_results(results: List[BenchmarkResult]) -> BenchmarkSummary:
    """Create summary from benchmark results."""
    if not results:
        return BenchmarkSummary(
            num_requests=0, total_tokens=0, total_time_s=0,
            ttft_min=0, ttft_p50=0, ttft_p95=0, ttft_p99=0, ttft_max=0,
            e2e_min=0, e2e_p50=0, e2e_p95=0, e2e_p99=0, e2e_max=0,
            throughput_tok_s=0, throughput_req_s=0, avg_output_tok_s=0,
            results=[]
        )
    
    ttfts = [r.ttft_ms for r in results]
    e2es = [r.e2e_ms for r in results]
    output_speeds = [r.tokens_per_second for r in results if r.tokens_per_second > 0]
    
    total_tokens = sum(r.generated_tokens for r in results)
    total_time = max(r.end_time for r in results) - min(r.submit_time for r in results)
    
    return BenchmarkSummary(
        num_requests=len(results),
        total_tokens=total_tokens,
        total_time_s=total_time,
        ttft_min=min(ttfts),
        ttft_p50=percentile(ttfts, 50),
        ttft_p95=percentile(ttfts, 95),
        ttft_p99=percentile(ttfts, 99),
        ttft_max=max(ttfts),
        e2e_min=min(e2es),
        e2e_p50=percentile(e2es, 50),
        e2e_p95=percentile(e2es, 95),
        e2e_p99=percentile(e2es, 99),
        e2e_max=max(e2es),
        throughput_tok_s=total_tokens / total_time if total_time > 0 else 0,
        throughput_req_s=len(results) / total_time if total_time > 0 else 0,
        avg_output_tok_s=statistics.mean(output_speeds) if output_speeds else 0,
        results=results,
    )


class DirectVLLMBenchmark:
    """Direct vLLM benchmark with true streaming TTFT measurement.
    
    FIXED: Now uses AsyncLLMEngine for true streaming TTFT measurement.
    Previously used sync API which could only estimate TTFT.
    """
    
    def __init__(
        self,
        model_id: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.7,  # Lower to allow for memory fragmentation
        use_async: bool = True,  # Use async engine for true TTFT (default: True)
    ):
        self.model_id = model_id
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_async = use_async
        self.engine = None
        self.async_engine = None
        self.tokenizer = None
        self._loop = None

    def _get_loop(self):
        """Return a persistent event loop reused across all async runs.

        CRITICAL: AsyncLLMEngine binds its background output-handler task to
        the loop that first drives it. Creating (and closing) a fresh loop per
        call orphans that task and the engine hangs forever on the next call.
        Cache one loop on the instance and reuse it for the engine's lifetime.
        """
        import asyncio
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            pass
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop
    
    def _init_engine(self):
        """Initialize vLLM engine (async for true TTFT, sync as fallback)."""
        if self.use_async:
            if self.async_engine is not None:
                return
            
            # Use AsyncLLMEngine for TRUE streaming TTFT measurement
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from transformers import AutoTokenizer
            
            print(f"Initializing AsyncLLMEngine with {self.model_id} (TRUE streaming TTFT)...")
            engine_args = AsyncEngineArgs(
                model=self.model_id,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                enable_prefix_caching=True,
            )
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            print("AsyncLLMEngine initialized (TRUE streaming TTFT enabled).")
        else:
            if self.engine is not None:
                return
            
            # Fallback to sync API (estimated TTFT only)
            from vllm import LLM
            print(f"Initializing vLLM sync engine with {self.model_id} (ESTIMATED TTFT)...")
            self.engine = LLM(
                model=self.model_id,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                enable_prefix_caching=True,
            )
            self.tokenizer = self.engine.get_tokenizer()
            print("vLLM sync engine initialized (TTFT will be ESTIMATED).")
    
    async def _run_single_request_async(
        self,
        req: BenchmarkRequest,
    ) -> BenchmarkResult:
        """Run a single request with TRUE streaming TTFT measurement."""
        from vllm import SamplingParams
        import uuid
        
        prompt_tokens = len(self.tokenizer.encode(req.prompt))
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=req.max_new_tokens,
        )
        
        request_id = str(uuid.uuid4())
        
        submit_time = time.perf_counter()
        first_token_time = None
        generated_tokens = 0
        
        # Stream tokens and measure TRUE TTFT
        async for output in self.async_engine.generate(
            req.prompt,
            sampling_params,
            request_id=request_id,
        ):
            if first_token_time is None and output.outputs[0].token_ids:
                first_token_time = time.perf_counter()
            generated_tokens = len(output.outputs[0].token_ids)
        
        end_time = time.perf_counter()
        
        # If no tokens were generated, set first_token_time to end_time
        if first_token_time is None:
            first_token_time = end_time
        
        return BenchmarkResult(
            request_id=req.request_id,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            max_new_tokens=req.max_new_tokens,
            submit_time=submit_time,
            first_token_time=first_token_time,  # ✅ TRUE TTFT (measured, not estimated)
            end_time=end_time,
        )
    
    def run_sequential(
        self,
        requests: List[BenchmarkRequest],
    ) -> BenchmarkSummary:
        """Run requests sequentially with TRUE streaming TTFT measurement.
        
        FIXED: Now uses AsyncLLMEngine for accurate TTFT measurement.
        Previously estimated TTFT using sync API.
        """
        import asyncio
        
        if self.use_async:
            # Use async engine for TRUE streaming TTFT
            self._init_engine()
            
            async def run_all():
                results = []
                print(f"Running {len(requests)} requests sequentially with TRUE streaming TTFT...")
                
                for i, req in enumerate(requests):
                    result = await self._run_single_request_async(req)
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Completed {i + 1}/{len(requests)}, TTFT: {result.ttft_ms:.1f}ms")
                
                return results
            
            # Run async function on the persistent loop (do NOT close it: the
            # engine's background tasks must survive for later concurrent runs)
            loop = self._get_loop()
            results = loop.run_until_complete(run_all())
            
            return summarize_results(results)
        else:
            # Fallback to sync API (estimated TTFT) - kept for compatibility
            self._init_engine()
            from vllm import SamplingParams
            
            results = []
            print(f"Running {len(requests)} requests sequentially (ESTIMATED TTFT)...")
            
            for i, req in enumerate(requests):
                prompt_tokens = len(self.tokenizer.encode(req.prompt))
                
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=req.max_new_tokens,
                )
                
                submit_time = time.perf_counter()
                outputs = self.engine.generate([req.prompt], sampling_params)
                end_time = time.perf_counter()
                
                output = outputs[0]
                generated_tokens = len(output.outputs[0].token_ids)
                
                # Estimate TTFT (sync API limitation)
                prefill_ratio = prompt_tokens / (prompt_tokens + generated_tokens)
                total_time = end_time - submit_time
                estimated_ttft = submit_time + (total_time * prefill_ratio * 0.1)
                
                result = BenchmarkResult(
                    request_id=req.request_id,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    max_new_tokens=req.max_new_tokens,
                    submit_time=submit_time,
                    first_token_time=estimated_ttft,  # ⚠️ ESTIMATED (sync API limitation)
                    end_time=end_time,
                )
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{len(requests)}")
            
            return summarize_results(results)
    
    def run_batched(
        self,
        requests: List[BenchmarkRequest],
        batch_size: int = 8,
    ) -> BenchmarkSummary:
        """Run requests in batches - measures continuous batching throughput.
        
        NOTE: Batched mode uses sync API, so TTFT is estimated.
        For true per-request TTFT in batches, use AsyncVLLMBenchmark.run_concurrent_async().
        """
        if self.use_async:
            print("⚠️  Batched mode not supported with async engine.")
            print("    Use run_concurrent() or AsyncVLLMBenchmark.run_concurrent_async() for true TTFT.")
            print("    Falling back to sync engine for batched mode...")
            self.use_async = False
        
        self._init_engine()
        from vllm import SamplingParams
        
        results = []
        print(f"Running {len(requests)} requests in batches of {batch_size} (ESTIMATED TTFT)...")
        
        # Process in batches
        for batch_start in range(0, len(requests), batch_size):
            batch = requests[batch_start:batch_start + batch_size]
            
            prompts = [r.prompt for r in batch]
            max_tokens = max(r.max_new_tokens for r in batch)
            
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
            )
            
            submit_time = time.perf_counter()
            outputs = self.engine.generate(prompts, sampling_params)
            end_time = time.perf_counter()
            
            # Record results for each request in batch
            for i, (req, output) in enumerate(zip(batch, outputs)):
                prompt_tokens = len(output.prompt_token_ids)
                generated_tokens = len(output.outputs[0].token_ids)
                
                # NOTE: TTFT is ESTIMATED in batched mode (sync API limitation)
                # In batched mode, all requests start together
                # TTFT is approximately when prefill completes for all
                # This is batch TTFT, not individual TTFT
                batch_ttft = submit_time + (end_time - submit_time) * 0.05  # ~5% for prefill (ESTIMATED)
                
                result = BenchmarkResult(
                    request_id=req.request_id,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=min(generated_tokens, req.max_new_tokens),
                    max_new_tokens=req.max_new_tokens,
                    submit_time=submit_time,
                    first_token_time=batch_ttft,  # ⚠️ ESTIMATED (batched mode limitation)
                    end_time=end_time,
                )
                results.append(result)
        
        return summarize_results(results)
    
    def run_concurrent(
        self,
        requests: List[BenchmarkRequest],
        max_concurrent: int = 16,
    ) -> BenchmarkSummary:
        """Run all requests concurrently with TRUE streaming TTFT.
        
        FIXED: Now uses async engine for true per-request TTFT measurement.
        Previously estimated TTFT using sync API.
        """
        import asyncio
        
        if self.use_async:
            # Use async engine for TRUE streaming TTFT
            self._init_engine()
            
            async def run_all():
                print(f"Running {len(requests)} requests concurrently with TRUE streaming TTFT (max {max_concurrent})...")
                
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def run_with_semaphore(req: BenchmarkRequest) -> BenchmarkResult:
                    async with semaphore:
                        return await self._run_single_request_async(req)
                
                # Submit all requests
                tasks = [run_with_semaphore(req) for req in requests]
                results = await asyncio.gather(*tasks)
                
                return list(results)
            
            # Reuse the persistent loop so the engine's background tasks
            # (started during run_sequential) remain alive and keep pumping.
            loop = self._get_loop()
            results = loop.run_until_complete(run_all())
            
            return summarize_results(results)
        else:
            # Fallback to sync API (estimated TTFT) - kept for compatibility
            self._init_engine()
            from vllm import SamplingParams
            
            print(f"Running {len(requests)} requests concurrently (ESTIMATED TTFT)...")
            
            prompts = [r.prompt for r in requests]
            max_tokens = max(r.max_new_tokens for r in requests)
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
            )
            
            submit_time = time.perf_counter()
            outputs = self.engine.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
            end_time = time.perf_counter()
            
            results = []
            for req, output in zip(requests, outputs):
                prompt_tokens = len(output.prompt_token_ids)
                generated_tokens = len(output.outputs[0].token_ids)
                
                # NOTE: TTFT is ESTIMATED in sync mode (limitation of sync API)
                idx = requests.index(req)
                progress = (idx + 1) / len(requests)
                estimated_ttft = submit_time + (end_time - submit_time) * 0.1 * progress
                
                result = BenchmarkResult(
                    request_id=req.request_id,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=min(generated_tokens, req.max_new_tokens),
                    max_new_tokens=req.max_new_tokens,
                    submit_time=submit_time,
                    first_token_time=estimated_ttft,  # ⚠️ ESTIMATED (sync API limitation)
                    end_time=end_time,
                )
                results.append(result)
            
            return summarize_results(results)


class DirectHFBenchmark:
    """Direct HuggingFace benchmark for comparison."""
    
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def _init_model(self):
        """Initialize HF model."""
        if self.model is not None:
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading HF model {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("HF model loaded.")
    
    def run_sequential(
        self,
        requests: List[BenchmarkRequest],
    ) -> BenchmarkSummary:
        """Run requests sequentially - HF baseline."""
        self._init_model()
        import torch
        
        results = []
        print(f"Running {len(requests)} requests sequentially on HF...")
        
        for i, req in enumerate(requests):
            inputs = self.tokenizer(
                req.prompt,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            prompt_tokens = inputs.input_ids.shape[1]
            
            submit_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            end_time = time.perf_counter()
            
            generated_tokens = outputs.shape[1] - prompt_tokens
            
            # HF generate is blocking - no streaming by default
            # TTFT = time to complete all generation (no continuous batching)
            # For fair comparison, we report actual completion time
            result = BenchmarkResult(
                request_id=req.request_id,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                max_new_tokens=req.max_new_tokens,
                submit_time=submit_time,
                first_token_time=end_time,  # HF: TTFT = E2E (no streaming)
                end_time=end_time,
            )
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(requests)}")
        
        return summarize_results(results)
