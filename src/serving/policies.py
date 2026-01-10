"""Dispatch policies for request scheduling."""

import time
from typing import List, Optional, Deque
from collections import deque
from dataclasses import dataclass


@dataclass
class PendingRequest:
    """A pending request in the queue."""
    request_id: str
    prompt: str
    max_new_tokens: int
    arrival_time: float
    dispatch_time: Optional[float] = None


class DispatchPolicy:
    """Base dispatch policy."""
    
    def should_dispatch(
        self,
        pending: Deque[PendingRequest],
        active_count: int,
        max_batch_size: int
    ) -> bool:
        """Check if we should dispatch now."""
        raise NotImplementedError
    
    def select_batch(
        self,
        pending: Deque[PendingRequest],
        max_batch_size: int,
        active_count: int
    ) -> List[PendingRequest]:
        """Select requests to dispatch."""
        raise NotImplementedError
    
    def has_pending(self) -> bool:
        """Check if policy has internally buffered requests.
        
        Policies that buffer requests (e.g., LengthAwareMicrobatchPolicy)
        must override this to return True when internal buffers are non-empty.
        This is CRITICAL for the main loop to not exit early.
        """
        return False


class FillBatchPolicy(DispatchPolicy):
    """Wait until batch is full."""
    
    def should_dispatch(self, pending, active_count, max_batch_size):
        if not pending:
            return False
        if active_count >= max_batch_size:
            return False
        return len(pending) >= max_batch_size or (pending and active_count == 0)
    
    def select_batch(self, pending, max_batch_size, active_count):
        batch = []
        available = max_batch_size - active_count
        while pending and len(batch) < available:
            batch.append(pending.popleft())
        return batch


class PeriodicPolicy(DispatchPolicy):
    """Dispatch every T milliseconds."""
    
    def __init__(self, interval_ms: float):
        self.interval_ms = interval_ms / 1000.0  # Convert to seconds
        self.last_dispatch = 0.0
    
    def should_dispatch(self, pending, active_count, max_batch_size):
        if not pending:
            return False
        if active_count >= max_batch_size:
            return False
        
        now = time.perf_counter()
        if now - self.last_dispatch >= self.interval_ms:
            self.last_dispatch = now
            return True
        return False
    
    def select_batch(self, pending, max_batch_size, active_count):
        batch = []
        available = max_batch_size - active_count
        while pending and len(batch) < available:
            batch.append(pending.popleft())
        return batch


class MaxWaitPolicy(DispatchPolicy):
    """Dispatch if oldest request waited > W ms."""
    
    def __init__(self, max_wait_ms: float):
        self.max_wait_ms = max_wait_ms / 1000.0
    
    def should_dispatch(self, pending, active_count, max_batch_size):
        if not pending:
            return False
        if active_count >= max_batch_size:
            return False
        
        oldest = pending[0]
        wait_time = time.perf_counter() - oldest.arrival_time
        return wait_time >= self.max_wait_ms
    
    def select_batch(self, pending, max_batch_size, active_count):
        batch = []
        available = max_batch_size - active_count
        while pending and len(batch) < available:
            batch.append(pending.popleft())
        return batch


class ShortFirstPolicy(DispatchPolicy):
    """Prioritize shorter requests."""
    
    def should_dispatch(self, pending, active_count, max_batch_size):
        if not pending:
            return False
        if active_count >= max_batch_size:
            return False
        return True
    
    def select_batch(self, pending, max_batch_size, active_count):
        # Sort by max_new_tokens (shortest first)
        sorted_pending = sorted(pending, key=lambda r: r.max_new_tokens)
        pending.clear()
        pending.extend(sorted_pending)
        
        batch = []
        available = max_batch_size - active_count
        while pending and len(batch) < available:
            batch.append(pending.popleft())
        return batch


class LengthAwareMicrobatchPolicy(DispatchPolicy):
    """Length-aware batching with microbatch window."""
    
    def __init__(self, window_ms: float, length_buckets: dict):
        self.window_ms = window_ms / 1000.0
        self.length_buckets = length_buckets
        self.last_dispatch = 0.0
        self.buckets = {
            "short": deque(),
            "medium": deque(),
            "long": deque(),
        }
    
    def has_pending(self) -> bool:
        """Check if any internal bucket has pending requests.
        
        CRITICAL: This method must return True if any requests are buffered,
        otherwise the main loop will exit early and drop requests.
        """
        return any(len(q) > 0 for q in self.buckets.values())
    
    def _bucket_request(self, req: PendingRequest) -> str:
        """Assign request to bucket."""
        if req.max_new_tokens <= self.length_buckets.get("short", 64):
            return "short"
        elif req.max_new_tokens <= self.length_buckets.get("medium", 128):
            return "medium"
        else:
            return "long"
    
    def should_dispatch(self, pending, active_count, max_batch_size):
        if not pending:
            # Check buckets
            total_bucketed = sum(len(q) for q in self.buckets.values())
            if total_bucketed == 0:
                return False
        if active_count >= max_batch_size:
            return False
        
        # Check if window expired
        now = time.perf_counter()
        if now - self.last_dispatch >= self.window_ms:
            return True
        
        # Or if any bucket has enough for a batch
        total_available = sum(len(q) for q in self.buckets.values()) + len(pending)
        if total_available >= max_batch_size - active_count:
            return True
        
        return False
    
    def select_batch(self, pending, max_batch_size, active_count):
        # Re-bucket any new pending requests
        while pending:
            req = pending.popleft()
            bucket = self._bucket_request(req)
            self.buckets[bucket].append(req)
        
        batch = []
        available = max_batch_size - active_count
        
        # Prioritize short, then medium, then long
        for bucket_name in ["short", "medium", "long"]:
            bucket_queue = self.buckets[bucket_name]
            while bucket_queue and len(batch) < available:
                batch.append(bucket_queue.popleft())
        
        self.last_dispatch = time.perf_counter()
        return batch
