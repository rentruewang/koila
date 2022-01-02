from typing import Protocol

from .datatypes import DataType
from .multidim import MultiDimensional


class MemoryInfo(MultiDimensional, DataType, Protocol):
    pass
