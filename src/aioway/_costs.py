# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from typing import Self

__all__ = ["DenseMeta"]


@dcls.dataclass
class DenseMeta:
    """
    Todo:
        Rewrite these similar ops into kernels.

    Todo:
        For now, use the word size as the cost, try to change it.
    """

    shape: tuple[int, ...]
    dtype: str
    cost: int

    def agg(self, *dims: int) -> Self:
        raise NotImplementedError

    def bcast(self, other: Self) -> Self:
        raise NotImplementedError

    def matmul(self, other: Self) -> Self:
        return type(self)(
            shape=compute_matmul_shape(self.shape, other.shape),
            dtype=promote_dtype(self.dtype, other.dtype),
            cost=compute_matmul_cost(self, other),
        )


def dtype_word_size(dtype: str):
    match dtype:
        case "int" | "int64" | "float" | "float64":
            return 64
        case "int32" | "float32":
            return 32
        case "int16" | "float16":
            return 16
        case "int8" | "char" | "bool":
            return 8
    raise NotImplementedError


def is_float(dtype: str):
    return "float" in dtype


def promote_dtype(self_dtype: str, other_dtype: str) -> str:
    bigger_word_dtype = max(dtype_word_size(self_dtype), dtype_word_size(other_dtype))
    promote_float = is_float(self_dtype) or is_float(other_dtype)

    int_or_float = "float" if promote_float else "int"
    return f"{int_or_float}{bigger_word_dtype}"


def compute_matmul_shape(
    self_shape: tuple[int, ...], other_shape: tuple[int, ...]
) -> tuple[int, ...]:
    # Only support 2 dimension for now.
    self_i, self_j = self_shape
    other_i, other_j = other_shape

    if self_j != other_i:
        raise ValueError

    return self_i, other_j


def compute_matmul_cost(self: DenseMeta, other: DenseMeta) -> int:
    output_dtype = promote_dtype(self.dtype, other.dtype)
    # Check if 2 dim, also compute output shape,
    i, j = self.shape
    _, k = other.shape
    return dtype_word_size(output_dtype) * i * j * k
