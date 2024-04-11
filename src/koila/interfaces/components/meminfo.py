from typing import Protocol

from .datatypes import DataType
from .multidim import MultiDimensional


class MemoryInfo(MultiDimensional, DataType, Protocol):
    """
    `MemoryInfo` contains all information needed to compute a tensor's memory usage.
    `MultiDimensional` is used to compute the number of elements,
    and `DataType` is used to compute the memory used per element.
    """
