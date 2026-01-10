"""Profiling utilities."""

import torch
from typing import Dict, Optional
from contextlib import contextmanager
import json


class Profiler:
    """Profiler for capturing performance data."""
    
    def __init__(self, use_torch_profiler: bool = True):
        self.use_torch_profiler = use_torch_profiler
        self.profiles = []
    
    @contextmanager
    def profile(self, name: str):
        """Profile a code block."""
        if self.use_torch_profiler and torch.cuda.is_available():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                yield prof
                
                # Extract key metrics
                events = prof.profiler.kineto_results.events()
                
                # Summarize
                summary = {
                    "name": name,
                    "cuda_time_ms": 0.0,
                    "cpu_time_ms": 0.0,
                    "memory_allocated_mb": 0.0,
                }
                
                for event in events:
                    if hasattr(event, "cuda_time"):
                        summary["cuda_time_ms"] += event.cuda_time / 1000.0
                    if hasattr(event, "cpu_time"):
                        summary["cpu_time_ms"] += event.cpu_time / 1000.0
                
                if torch.cuda.is_available():
                    summary["memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                
                self.profiles.append(summary)
        else:
            # Fallback: just time it
            import time
            start = time.perf_counter()
            yield None
            elapsed = (time.perf_counter() - start) * 1000
            
            self.profiles.append({
                "name": name,
                "elapsed_ms": elapsed,
            })
    
    def get_summary(self) -> Dict:
        """Get profiling summary."""
        if not self.profiles:
            return {}
        
        total_cuda = sum(p.get("cuda_time_ms", 0) for p in self.profiles)
        total_cpu = sum(p.get("cpu_time_ms", 0) for p in self.profiles)
        max_memory = max((p.get("memory_allocated_mb", 0) for p in self.profiles), default=0)
        
        return {
            "total_cuda_time_ms": total_cuda,
            "total_cpu_time_ms": total_cpu,
            "max_memory_mb": max_memory,
            "profiles": self.profiles,
        }
    
    def save(self, filepath: str):
        """Save profiling data."""
        from ..utils.io import save_json
        data = self.get_summary()
        save_json(data, filepath)
