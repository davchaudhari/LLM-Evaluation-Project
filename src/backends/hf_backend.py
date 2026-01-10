"""HuggingFace Transformers backend."""

import torch
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .backend_base import BackendBase, GenerationRequest, GenerationResult


class HuggingFaceBackend(BackendBase):
    """HuggingFace Transformers backend (sequential processing)."""
    
    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        device: str = "cuda"
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        
        print(f"Loading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else device,
            trust_remote_code=True
        )
        self.model.eval()
        
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
        }
    
    def generate(
        self,
        requests: List[GenerationRequest],
        stream: bool = False
    ) -> List[GenerationResult]:
        """Generate for requests (sequential, not true batching)."""
        results = []
        
        for req in requests:
            inputs = self.tokenizer(
                req.prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            prompt_tokens = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            generated_ids = outputs[0][prompt_tokens:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            results.append(GenerationResult(
                request_id=req.request_id,
                generated_text=generated_text,
                generated_tokens=len(generated_ids),
                prompt_tokens=prompt_tokens,
            ))
            
            self._stats["total_requests"] += 1
            self._stats["total_tokens"] += len(generated_ids)
        
        return results
    
    async def generate_stream(self, request: GenerationRequest):
        """Stream generation (not implemented for HF)."""
        # For HF, we'd need TextIteratorStreamer
        # For now, just return full result
        results = self.generate([request])
        text = results[0].generated_text
        for char in text:
            yield char
    
    def get_stats(self) -> dict:
        """Get backend statistics."""
        return self._stats.copy()
