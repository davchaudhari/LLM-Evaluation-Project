"""Base backend interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, AsyncIterator
from dataclasses import dataclass


@dataclass
class GenerationRequest:
    """A generation request."""
    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class GenerationResult:
    """Result of generation."""
    request_id: str
    generated_text: str
    generated_tokens: int
    prompt_tokens: int
    finish_reason: str = "length"


class BackendBase(ABC):
    """Base class for serving backends."""
    
    @abstractmethod
    def generate(
        self,
        requests: List[GenerationRequest],
        stream: bool = False
    ) -> List[GenerationResult]:
        """Generate for a batch of requests."""
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Stream tokens for a single request."""
        pass
    
    @abstractmethod
    def get_stats(self) -> dict:
        """Get backend statistics."""
        pass
