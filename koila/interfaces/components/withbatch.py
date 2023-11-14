from __future__ import annotations

import dataclasses as dcls
from typing import Protocol


@dcls.dataclass
class BatchInfo:
    """
    `BatchInfo` contains information that marks a dimension as splittable.
    """

    dimension: int
    "The dimension (which axis) of the batch."

    size: int
    "The size (number of entries) of the batch."


class WithBatch(Protocol):
    """
    `WithBatch` marks a tensor's ability to be splitted across a special dimension, batch.
    Since `Koila`'s ability's to accumulate gradients across batches,
    the batch dimension marked by this protocol should contain entries independent to one another/

    For example, a set of images can be shuffled, but individual pixels within an image cannot not.
    Thereforce, the images can be marked as in a batch, but pixels cannot.
    """

    batch: BatchInfo | None
