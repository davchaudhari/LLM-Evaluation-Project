"""Streaming benchmark with true TTFT measurement.

Uses vLLM's AsyncLLMEngine to measure actual time to first token.
"""

import time
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional
import uuid

from .direct_benchmark import (
    BenchmarkRequest,
    BenchmarkResult,
    BenchmarkSummary,
    summarize_results,
)


class AsyncVLLMBenchmark:
    """Async vLLM benchmark with TRUE streaming TTFT measurement."""
    
    def __init__(
        self,
        model_id: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.7,  # Lower to allow for memory fragmentation
    ):
        self.model_id = model_id
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.engine = None
        self.tokenizer = None
    
    async def _init_engine(self):
        """Initialize async vLLM engine."""
        if self.engine is not None:
            return
        
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from transformers import AutoTokenizer
        
        print(f"Initializing AsyncLLMEngine with {self.model_id}...")
        
        engine_args = AsyncEngineArgs(
            model=self.model_id,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=True,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        print("AsyncLLMEngine initialized.")
    
    async def _run_single_request(
        self,
        request: BenchmarkRequest,
    ) -> BenchmarkResult:
        """Run a single request with streaming TTFT measurement."""
        from vllm import SamplingParams
        
        prompt_tokens = len(self.tokenizer.encode(request.prompt))
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=request.max_new_tokens,
        )
        
        request_id = str(uuid.uuid4())
        
        submit_time = time.perf_counter()
        first_token_time = None
        generated_tokens = 0
        
        # Stream tokens and measure TTFT
        async for output in self.engine.generate(
            request.prompt,
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
            request_id=request.request_id,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            max_new_tokens=request.max_new_tokens,
            submit_time=submit_time,
            first_token_time=first_token_time,
            end_time=end_time,
        )
    
    async def run_sequential_async(
        self,
        requests: List[BenchmarkRequest],
    ) -> BenchmarkSummary:
        """Run requests sequentially with true streaming TTFT."""
        await self._init_engine()
        
        results = []
        print(f"Running {len(requests)} requests sequentially with streaming...")
        
        for i, req in enumerate(requests):
            result = await self._run_single_request(req)
            results.append(result)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{len(requests)}, TTFT: {result.ttft_ms:.1f}ms, E2E: {result.e2e_ms:.1f}ms")
        
        return summarize_results(results)
    
    async def run_concurrent_async(
        self,
        requests: List[BenchmarkRequest],
        max_concurrent: int = 16,
    ) -> BenchmarkSummary:
        """Run requests concurrently with true streaming TTFT.
        
        This simulates real serving load with concurrent requests.
        """
        await self._init_engine()
        
        print(f"Running {len(requests)} requests concurrently (max {max_concurrent})...")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(req: BenchmarkRequest) -> BenchmarkResult:
            async with semaphore:
                return await self._run_single_request(req)
        
        # Submit all requests
        tasks = [run_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        return summarize_results(list(results))
    
    async def run_arrival_pattern(
        self,
        requests: List[BenchmarkRequest],
        arrival_times: List[float],  # Relative times in seconds
        max_concurrent: int = 32,
    ) -> BenchmarkSummary:
        """Run requests with specified arrival pattern.
        
        Simulates realistic arrival patterns (burst, Poisson, etc.)
        """
        await self._init_engine()
        
        print(f"Running {len(requests)} requests with arrival pattern...")
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_at_time(
            req: BenchmarkRequest,
            delay: float,
        ) -> BenchmarkResult:
            if delay > 0:
                await asyncio.sleep(delay)
            async with semaphore:
                return await self._run_single_request(req)
        
        # Schedule all requests according to arrival times
        start_time = time.perf_counter()
        tasks = []
        
        for req, arrival in zip(requests, arrival_times):
            task = asyncio.create_task(run_at_time(req, arrival))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        return summarize_results(list(results))
    
    def _get_or_create_loop(self):
        """Get existing event loop or create new one."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    def run_sequential(self, requests: List[BenchmarkRequest]) -> BenchmarkSummary:
        """Sync wrapper for sequential runs."""
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self.run_sequential_async(requests))
    
    def run_concurrent(
        self,
        requests: List[BenchmarkRequest],
        max_concurrent: int = 16,
    ) -> BenchmarkSummary:
        """Sync wrapper for concurrent runs - reuses existing event loop."""
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self.run_concurrent_async(requests, max_concurrent))


def generate_workload(
    num_requests: int,
    prompt_tokens: int = 128,
    max_new_tokens: int = 64,
    prompt_template: str = "Write a detailed explanation of {topic}. Be thorough.",
) -> List[BenchmarkRequest]:
    """Generate a standardized workload."""
    topics = [
        "machine learning", "quantum computing", "climate change",
        "artificial intelligence", "blockchain technology", "renewable energy",
        "space exploration", "genetic engineering", "cybersecurity",
        "neural networks", "data science", "robotics", "cloud computing",
        "virtual reality", "internet of things", "5G technology",
    ]
    
    requests = []
    for i in range(num_requests):
        topic = topics[i % len(topics)]
        prompt = prompt_template.format(topic=topic)
        
        # Pad prompt to approximate target token count
        while len(prompt.split()) < prompt_tokens // 2:
            prompt = prompt + f" Consider the implications of {topic}."
        
        requests.append(BenchmarkRequest(
            request_id=f"req_{i:04d}",
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        ))
    
    return requests


def generate_mixed_workload(
    num_requests: int,
    short_ratio: float = 0.7,
    short_tokens: int = 32,
    long_tokens: int = 256,
) -> List[BenchmarkRequest]:
    """Generate mixed short/long workload for HOL blocking test."""
    import random
    
    requests = []
    for i in range(num_requests):
        is_short = random.random() < short_ratio
        max_tokens = short_tokens if is_short else long_tokens
        
        prompt = f"Request {i}: Generate a response."
        
        requests.append(BenchmarkRequest(
            request_id=f"req_{i:04d}_{'short' if is_short else 'long'}",
            prompt=prompt,
            max_new_tokens=max_tokens,
        ))
    
    return requests
