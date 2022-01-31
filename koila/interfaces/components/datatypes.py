from __future__ import annotations

from typing import Protocol

from torch import device as Device
from torch import dtype as DType


class DataType(Protocol):
    """
    `DataType` represents a tensor's datatype.
    A tensor's datatype has two attributes we care about,
    its width in memory (`dtype`), and the device it's on (`device`).
    """
    
    dtype: DType
    "The datatype of a tensor."

    device: str | Device
    "The device the tensor is on."
