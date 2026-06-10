"""Workload generation for experiments."""

import random
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class WorkloadRequest:
    """A request in a workload."""
    request_id: str
    prompt: str
    max_new_tokens: int
    arrival_time: float = 0.0


class WorkloadGenerator:
    """Generate workloads for experiments."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.prompts = [
            "What is machine learning?",
            "Explain Python programming.",
            "What is a neural network?",
            "Define artificial intelligence.",
            "What is deep learning?",
            "Explain cloud computing.",
            "What is a database?",
            "Define an API.",
            "What is quantum computing?",
            "Explain blockchain technology.",
        ]
    
    def generate_burst(
        self,
        num_requests: int,
        max_tokens: int = 64,
        base_time: float = 0.0
    ) -> List[WorkloadRequest]:
        """Generate burst workload (all at once)."""
        requests = []
        for i in range(num_requests):
            requests.append(WorkloadRequest(
                request_id=f"req_{i:04d}",
                prompt=self.prompts[i % len(self.prompts)],
                max_new_tokens=max_tokens,
                arrival_time=base_time,
            ))
        return requests
    
    def generate_poisson(
        self,
        num_requests: int,
        lambda_rate: float,
        max_tokens: int = 64,
        base_time: float = 0.0
    ) -> List[WorkloadRequest]:
        """Generate Poisson arrival workload."""
        requests = []
        cumulative_time = 0.0
        
        for i in range(num_requests):
            if i > 0:
                inter_arrival = random.expovariate(lambda_rate)
                cumulative_time += inter_arrival
            
            requests.append(WorkloadRequest(
                request_id=f"req_{i:04d}",
                prompt=self.prompts[i % len(self.prompts)],
                max_new_tokens=max_tokens,
                arrival_time=base_time + cumulative_time,
            ))
        
        return requests
    
    def generate_mixed_lengths(
        self,
        num_requests: int,
        short_ratio: float = 0.5,
        short_tokens: int = 32,
        long_tokens: int = 256,
        base_time: float = 0.0,
        lambda_rate: Optional[float] = None,
    ) -> List[WorkloadRequest]:
        """Generate mixed-length workload.

        For short_ratio=0.5, alternates between short and long to ensure exact ratio.
        For other ratios, uses random sampling.

        If ``lambda_rate`` (requests/sec) is given, requests are assigned
        Poisson (exponential inter-arrival) arrival times instead of all
        arriving at once. This produces a realistic load profile so that
        queueing/head-of-line effects reflect genuine contention rather than a
        single synchronized burst.
        """
        requests = []
        cumulative_time = 0.0
        for i in range(num_requests):
            if short_ratio == 0.5:
                # Exact 50/50 split by alternating
                is_short = (i % 2 == 0)
            else:
                # Random sampling for other ratios
                is_short = random.random() < short_ratio
            max_tokens = short_tokens if is_short else long_tokens

            if lambda_rate is not None and i > 0:
                cumulative_time += random.expovariate(lambda_rate)
            arrival = base_time + (cumulative_time if lambda_rate is not None else 0.0)

            requests.append(WorkloadRequest(
                request_id=f"req_{i:04d}",
                prompt=self.prompts[i % len(self.prompts)],
                max_new_tokens=max_tokens,
                arrival_time=arrival,
            ))
        return requests
