"""Timing utilities for precise latency measurement."""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimingStats:
    """Timing statistics for a request."""
    arrival_time: float = 0.0
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds."""
        if self.first_token_time and self.arrival_time:
            return (self.first_token_time - self.arrival_time) * 1000
        return 0.0
    
    @property
    def e2e_ms(self) -> float:
        """End-to-end latency in milliseconds."""
        if self.end_time and self.arrival_time:
            return (self.end_time - self.arrival_time) * 1000
        return 0.0
    
    @property
    def queue_time_ms(self) -> float:
        """Queue wait time (if dispatch_time is set)."""
        return 0.0  # Can be extended


def get_timestamp_ms() -> float:
    """Get current timestamp in milliseconds."""
    return time.perf_counter() * 1000


def get_timestamp() -> float:
    """Get current timestamp in seconds."""
    return time.perf_counter()
