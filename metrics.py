from __future__ import annotations
from dataclasses import dataclass
import time

@dataclass
class Metrics:
    elapsed_ms: int
    output_chars: int

class Timer:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._t1 = time.perf_counter()

    @property
    def elapsed_ms(self) -> int:
        return int((self._t1 - self._t0) * 1000)