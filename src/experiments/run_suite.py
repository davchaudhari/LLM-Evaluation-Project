"""Run experiment suite on a backend.

This module provides the ExperimentRunner class for running workloads
through different dispatch policies and collecting metrics.

IMPORTANT: This is SCHEDULER-LAYER measurement. For true backend TTFT,
use the direct benchmark (src/benchmarks/direct_benchmark.py).

Measurement layer: scheduler_layer
- Measures request flow through dispatch policies
- TTFT is approximate (batch start time, not true first token)
- Use for policy comparison experiments, not raw backend benchmarks
"""

import time
import asyncio
from typing import List, Optional
from collections import deque

from ..backends.backend_base import BackendBase, GenerationRequest
from ..serving.policies import DispatchPolicy, FillBatchPolicy
from .workloads import WorkloadGenerator, WorkloadRequest
from .metrics import MetricsAggregator, RequestMetrics
from ..utils.timing import get_timestamp


class ExperimentRunner:
    """Run experiments on a backend with scheduler-layer metrics.
    
    NOTE: This measures scheduler-layer timing, not true backend TTFT.
    For direct backend benchmarks, use DirectVLLMBenchmark or DirectHFBenchmark.
    """
    
    def __init__(
        self,
        backend: BackendBase,
        policy: DispatchPolicy,
        max_batch_size: int = 8,
        backend_name: str = "unknown"
    ):
        self.backend = backend
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.backend_name = backend_name
        self.metrics = MetricsAggregator(measurement_layer="scheduler_layer")
    
    def run_workload(
        self,
        workload: List[WorkloadRequest],
        policy_name: str = "unknown",
        use_streaming_ttft: bool = False,
        respect_arrival_times: bool = False,
    ) -> MetricsAggregator:
        """Run a workload and collect metrics.
        
        TIMING FIX: Workload arrival_time is relative (from base_time=0).
        We normalize to absolute timestamps at experiment start.
        
        BUG FIX: Loop now checks policy.has_pending() to handle policies
        that buffer requests internally (e.g., LengthAwareMicrobatchPolicy).

        ARRIVAL GATING: When ``respect_arrival_times`` is True, a request only
        becomes visible to the dispatch policy once wall-clock time reaches its
        absolute arrival time. This makes Poisson/open-loop workloads behave
        like real load (requests trickle in) instead of a synchronized burst
        where every request is queued at t=0. Default False preserves the
        original burst behavior for existing experiments.
        """
        self.metrics = MetricsAggregator(measurement_layer="scheduler_layer")
        
        if use_streaming_ttft and hasattr(self.backend, 'generate_stream'):
            # Use streaming for accurate TTFT
            return self.run_streaming_workload(workload, policy_name)
        
        # Track expected vs processed requests for validation
        expected_request_ids = {req.request_id for req in workload}
        processed_request_ids = set()
        
        # TIMING FIX: Normalize arrival times to absolute timestamps
        # Workload arrival_time is relative (e.g., 0.0 for burst).
        # We add the experiment start time to make them absolute.
        experiment_start = get_timestamp()
        arrival_offset = experiment_start
        
        # Create a mapping of request_id -> absolute arrival time
        arrival_times = {}
        for req in workload:
            # Convert relative arrival time to absolute
            arrival_times[req.request_id] = arrival_offset + req.arrival_time
        
        # Batch processing mode
        pending = deque()
        active_requests = {}

        # Requests not yet "arrived" (only used when respect_arrival_times=True),
        # sorted ascending by absolute arrival time for O(1) front admission.
        if respect_arrival_times:
            not_arrived = deque(
                sorted(workload, key=lambda r: arrival_times[r.request_id])
            )
        else:
            not_arrived = deque()
            # Burst mode: submit all requests immediately
            for req in workload:
                pending.append(req)
        
        # Process requests
        # BUG FIX: Added self.policy.has_pending() to handle policies with internal buffers
        while pending or active_requests or self.policy.has_pending() or not_arrived:
            # Admit any requests whose arrival time has now passed.
            if not_arrived:
                now = get_timestamp()
                while not_arrived and arrival_times[not_arrived[0].request_id] <= now:
                    pending.append(not_arrived.popleft())

            # Check dispatch policy
            if self.policy.should_dispatch(
                pending,
                len(active_requests),
                self.max_batch_size
            ):
                batch = self.policy.select_batch(
                    pending,
                    self.max_batch_size,
                    len(active_requests)
                )
                
                # Mark dispatch time
                dispatch_time = get_timestamp()
                for req in batch:
                    req.dispatch_time = dispatch_time
                    active_requests[req.request_id] = req
            
            if not active_requests:
                # BUG FIX: Also check policy.has_pending() before breaking.
                # With arrival gating, also wait while requests are still
                # scheduled to arrive in the future.
                if not pending and not self.policy.has_pending() and not not_arrived:
                    break
                time.sleep(0.001)
                continue
            
            # Process batch
            batch_requests = []
            batch_req_map = {}  # Map request_id to original req
            for req in list(active_requests.values()):
                gen_req = GenerationRequest(
                    request_id=req.request_id,
                    prompt=req.prompt,
                    max_new_tokens=req.max_new_tokens,
                )
                batch_requests.append(gen_req)
                batch_req_map[req.request_id] = req
            
            if batch_requests:
                # Generate
                start_time = get_timestamp()
                results = self.backend.generate(batch_requests)
                end_time = get_timestamp()
                
                # Record metrics
                for result in results:
                    req = active_requests.pop(result.request_id, None)
                    if req:
                        # Track processed requests
                        processed_request_ids.add(result.request_id)
                        
                        # Truncate generated tokens to original max_new_tokens if needed
                        actual_gen_tokens = min(result.generated_tokens, req.max_new_tokens)
                        
                        # Use absolute arrival time
                        abs_arrival_time = arrival_times.get(req.request_id, start_time)
                        
                        metric = RequestMetrics(
                            request_id=result.request_id,
                            prompt=req.prompt,
                            prompt_tokens=result.prompt_tokens,
                            generated_tokens=actual_gen_tokens,
                            max_new_tokens=req.max_new_tokens,
                            arrival_time=abs_arrival_time,
                            first_token_time=start_time,  # Approximate for batch
                            end_time=end_time,
                            dispatch_time=req.dispatch_time,
                            backend=self.backend_name,
                            policy=policy_name,
                            measurement_layer="scheduler_layer",
                        )
                        self.metrics.add(metric)
        
        # VALIDATION: Ensure all requests were processed
        if processed_request_ids != expected_request_ids:
            missing = expected_request_ids - processed_request_ids
            extra = processed_request_ids - expected_request_ids
            
            error_msg = (
                f"Request count mismatch! Expected {len(expected_request_ids)}, "
                f"processed {len(processed_request_ids)}. "
                f"Missing: {len(missing)}, Extra: {len(extra)}"
            )
            print(f"❌ CRITICAL: {error_msg}")
            
            # Dump debug info
            try:
                from pathlib import Path
                debug_dir = Path("/results/runs")
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_file = debug_dir / f"dropped_requests_{policy_name}.txt"
                with open(debug_file, "w") as f:
                    f.write(f"Policy: {policy_name}\n")
                    f.write(f"Expected: {len(expected_request_ids)}\n")
                    f.write(f"Processed: {len(processed_request_ids)}\n")
                    f.write(f"\nMissing request IDs:\n")
                    for rid in sorted(missing):
                        f.write(f"  {rid}\n")
                print(f"   Debug info saved to {debug_file}")
            except Exception as e:
                print(f"   Could not save debug file: {e}")
            
            raise RuntimeError(error_msg)
        
        # Validate metrics before returning
        validation = self.metrics.validate_all(fail_on_error=False)
        if not validation["valid"]:
            print(f"⚠️  Metrics validation issues: {validation}")
        
        return self.metrics
    
    def run_streaming_workload(
        self,
        workload: List[WorkloadRequest],
        policy_name: str = "unknown"
    ) -> MetricsAggregator:
        """Run workload with streaming TTFT measurement."""
        self.metrics = MetricsAggregator(measurement_layer="scheduler_layer")
        
        # TIMING FIX: Use experiment start as base for arrival times
        experiment_start = get_timestamp()
        
        # For streaming, process sequentially to measure real TTFT
        for req in workload:
            gen_req = GenerationRequest(
                request_id=req.request_id,
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
            )
            
            # Convert relative arrival time to absolute
            abs_arrival = experiment_start + req.arrival_time
            
            first_token = None
            tokens = []
            
            async def collect_tokens():
                nonlocal first_token, tokens
                async for token in self.backend.generate_stream(gen_req):
                    if token and first_token is None:
                        first_token = get_timestamp()
                    tokens.append(token)
            
            asyncio.run(collect_tokens())
            end_time = get_timestamp()
            
            metric = RequestMetrics(
                request_id=req.request_id,
                prompt=req.prompt,
                prompt_tokens=0,  # Would need tokenizer to count
                generated_tokens=len(''.join(tokens)),  # Approximate
                max_new_tokens=req.max_new_tokens,
                arrival_time=abs_arrival,
                first_token_time=first_token,
                end_time=end_time,
                dispatch_time=getattr(req, 'dispatch_time', None),
                backend=self.backend_name,
                policy=policy_name,
                measurement_layer="scheduler_layer",
            )
            self.metrics.add(metric)
        
        # Validate metrics before returning
        validation = self.metrics.validate_all(fail_on_error=False)
        if not validation["valid"]:
            print(f"⚠️  Metrics validation issues: {validation}")
        
        return self.metrics
