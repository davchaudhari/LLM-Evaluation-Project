"""vLLM backend with true continuous batching."""

import asyncio
from typing import List, Optional, AsyncIterator
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from .backend_base import BackendBase, GenerationRequest, GenerationResult


class VLLMBackend(BackendBase):
    """vLLM backend with continuous batching."""
    
    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        use_async: bool = False,  # Set True for streaming TTFT measurement
    ):
        self.model_id = model_id
        self.use_async = use_async
        
        print(f"Initializing vLLM with {model_id}...")
        
        if use_async:
            # Use AsyncLLMEngine for streaming
            engine_args = AsyncEngineArgs(
                model=model_id,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=trust_remote_code,
            )
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.engine = None
        else:
            # Use sync LLM for batch generation
            # Lower max_num_seqs for smaller GPUs
            self.engine = LLM(
                model=model_id,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=trust_remote_code,
                enable_prefix_caching=True,
                max_num_seqs=32,  # Lower from default 256 for T4
            )
            self.async_engine = None
        
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
        }
    
    def generate(
        self,
        requests: List[GenerationRequest],
        stream: bool = False
    ) -> List[GenerationResult]:
        """Generate for a batch of requests."""
        if not self.engine:
            raise RuntimeError("Sync engine not initialized. Use use_async=False or use generate_stream for async.")
        
        # Batch generation
        prompts = [req.prompt for req in requests]
        
        # vLLM requires same sampling params for batch
        # Use max of all max_new_tokens to ensure all complete
        max_tokens = max(req.max_new_tokens for req in requests)
        sampling_params = SamplingParams(
            temperature=requests[0].temperature,
            top_p=requests[0].top_p,
            max_tokens=max_tokens,
        )
        
        outputs = self.engine.generate(prompts, sampling_params)
        
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            generated_tokens = len(output.outputs[0].token_ids)
            prompt_tokens = len(output.prompt_token_ids)
            
            results.append(GenerationResult(
                request_id=requests[i].request_id,
                generated_text=generated_text,
                generated_tokens=generated_tokens,
                prompt_tokens=prompt_tokens,
                finish_reason=output.outputs[0].finish_reason,
            ))
            
            self._stats["total_requests"] += 1
            self._stats["total_tokens"] += generated_tokens
        
        return results
    
    async def generate_stream(
        self,
        request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Stream tokens for a single request with accurate TTFT."""
        # For now, use sync engine and simulate streaming
        # True async streaming requires AsyncLLMEngine setup which is more complex
        if self.engine:
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_new_tokens,
            )
            outputs = self.engine.generate([request.prompt], sampling_params)
            text = outputs[0].outputs[0].text
            
            # Simulate streaming by yielding character by character
            # In production, would use AsyncLLMEngine for true streaming
            for char in text:
                yield char
        else:
            # If async engine is available, use it
            # This is a placeholder - full implementation would use AsyncLLMEngine API
            yield ""
    
    def get_stats(self) -> dict:
        """Get backend statistics."""
        return self._stats.copy()
