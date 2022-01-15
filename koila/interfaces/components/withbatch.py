from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class BatchInfo:
    dimension: int
    size: int


class WithBatch(Protocol):
    batch: BatchInfo | None
