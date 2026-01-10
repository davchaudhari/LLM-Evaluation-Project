"""Scheduler for managing request queues and batching."""

import time
import threading
from typing import List, Optional, Deque
from collections import deque
from dataclasses import dataclass

from .policies import DispatchPolicy, PendingRequest


@dataclass
class ScheduledRequest:
    """A request scheduled for processing."""
    request_id: str
    prompt: str
    max_new_tokens: int
    arrival_time: float
    dispatch_time: Optional[float] = None


class Scheduler:
    """Manages request queue and dispatch."""
    
    def __init__(
        self,
        policy: DispatchPolicy,
        max_batch_size: int = 8
    ):
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.pending: Deque[PendingRequest] = deque()
        self.lock = threading.Lock()
    
    def submit(
        self,
        request_id: str,
        prompt: str,
        max_new_tokens: int,
        arrival_time: Optional[float] = None
    ):
        """Submit a request to the queue."""
        if arrival_time is None:
            arrival_time = time.perf_counter()
        
        req = PendingRequest(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            arrival_time=arrival_time,
        )
        
        with self.lock:
            self.pending.append(req)
    
    def get_next_batch(self, active_count: int) -> List[PendingRequest]:
        """Get next batch to dispatch."""
        with self.lock:
            if self.policy.should_dispatch(self.pending, active_count, self.max_batch_size):
                batch = self.policy.select_batch(
                    self.pending,
                    self.max_batch_size,
                    active_count
                )
                
                # Mark dispatch time
                dispatch_time = time.perf_counter()
                for req in batch:
                    req.dispatch_time = dispatch_time
                
                return batch
            return []
    
    def has_pending(self) -> bool:
        """Check if there are pending requests."""
        with self.lock:
            return len(self.pending) > 0
    
    def get_queue_length(self) -> int:
        """Get current queue length."""
        with self.lock:
            return len(self.pending)
