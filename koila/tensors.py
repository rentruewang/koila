from __future__ import annotations

import functools
import operator
from abc import abstractmethod
from typing import NoReturn, Protocol, Tuple, overload

from torch import device as Device
from torch import dtype as DType
from torch.functional import Tensor


class TensorLike(Protocol):
    dtype: DType
    device: Device

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
    def __divmod__(self, other: TensorLike) -> NoReturn:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    def dim(self) -> int:
        return len(self.size())

    @property
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
    def shape(self) -> Tuple[int, ...]:
        return self.size()

    def numel(self) -> int:
        return functools.reduce(operator.mul, self.size)

    @property
    @abstractmethod
    def T(self) -> TensorLike:
        ...

    def torch(self) -> Tensor:
        ...
