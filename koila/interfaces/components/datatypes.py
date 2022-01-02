from __future__ import annotations

from typing import Protocol

from torch import device as Device
from torch import dtype as DType


class DataType(Protocol):
    dtype: DType
    device: str | Device
