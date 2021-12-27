from __future__ import annotations

import functools
import operator
from abc import abstractmethod
from typing import Callable, NamedTuple, NoReturn, Protocol, Tuple, overload

from torch import Tensor
from torch import device as Device
from torch import dtype as DType

from .runnables import Runnable


class BatchedPair(NamedTuple):
    batch: int
    no_batch: int


class BatchInfo(NamedTuple):
    index: int
    value: int

    def map(self, func: Callable[[int], int]) -> BatchInfo:
        index = func(self.index)
        return BatchInfo(index, self.value)


class TensorLike(Runnable[Tensor], Protocol):
    """
    TensorLike is a protocol that mimics PyTorch's Tensor.
    """

    dtype: DType
    device: str | Device
    batch: BatchInfo | None

    @abstractmethod
    def run(self, partial: Tuple[int, int] | None = None) -> Tensor:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def __bool__(self) -> bool:
        ...

    @abstractmethod
    def __int__(self) -> int:
        ...

    @abstractmethod
    def __float__(self) -> float:
        ...

    @abstractmethod
    def __invert__(self) -> bool:
        ...

    @abstractmethod
    def __pos__(self) -> TensorLike:
        ...

    @abstractmethod
    def __neg__(self) -> TensorLike:
        ...

    @abstractmethod
    def __add__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __radd__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __sub__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __rsub__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __mul__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __rmul__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __truediv__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __rtruediv__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __floordiv__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __rfloordiv__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __pow__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __rpow__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __mod__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __rmod__(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def __divmod__(self, other: TensorLike) -> NoReturn:
        ...

    @abstractmethod
    def __rdivmod__(self, other: TensorLike) -> NoReturn:
        ...

    @abstractmethod
    def __abs__(self) -> TensorLike:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __matmul__(self, other: TensorLike) -> NoReturn:
        ...

    @abstractmethod
    def __rmatmul__(self, other: TensorLike) -> NoReturn:
        ...

    @abstractmethod
    def __eq__(self, other: TensorLike) -> NoReturn:
        ...

    @abstractmethod
    def __ne__(self, other: TensorLike) -> NoReturn:
        ...

    @abstractmethod
    def __gt__(self, other: TensorLike) -> NoReturn:
        ...

    @abstractmethod
    def __ge__(self, other: TensorLike) -> NoReturn:
        ...

    @abstractmethod
    def __lt__(self, other: TensorLike) -> NoReturn:
        ...

    @abstractmethod
    def __le__(self, other: TensorLike) -> NoReturn:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def add(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def sub(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def mul(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def div(self, other: TensorLike) -> TensorLike:
        ...

    divide = truediv = div

    @abstractmethod
    def pow(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def abs(self) -> TensorLike:
        ...

    @abstractmethod
    def matmul(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def eq(self, other: TensorLike) -> TensorLike:
        return not self.ne(other)

    @abstractmethod
    def ne(self, other: TensorLike) -> TensorLike:
        return not self.eq(other)

    @abstractmethod
    def gt(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def ge(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def lt(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def le(self, other: TensorLike) -> TensorLike:
        ...

    @abstractmethod
    def dim(self) -> int:
        return len(self.size())

    @property
    @abstractmethod
    def ndim(self) -> int:
        return self.dim()

    @overload
    @abstractmethod
    def size(self) -> Tuple[int, ...]:
        ...

    @overload
    @abstractmethod
    def size(self, dim: int) -> int:
        ...

    @abstractmethod
    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        ...

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        return self.size()

    @abstractmethod
    def numel(self) -> int:
        return functools.reduce(operator.mul, self.shape)

    @property
    @abstractmethod
    def T(self) -> TensorLike:
        ...
