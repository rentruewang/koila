from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Sequence, Tuple, Type

from rich.logging import RichHandler
from torch import Tensor
from torch import device as Device
from torch import dtype as DType

from .interfaces import BatchInfo, RunnableTensor, TensorLike

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(RichHandler())
LOGGER.setLevel(logging.DEBUG)

# So, it seems that torch's Tensor base class utilizes metaclass
# to pretend to be a parent of LongTensor, FloatTensor etc.
# Perhaps I'll be using the same paradigm.


class EagerTensor(RunnableTensor):
    def __init__(self, data: Tensor) -> None:
        self.data = data

    def __getattr__(self, name: str) -> Any:
        return getattr(self.data, name)

    def batch(self) -> BatchInfo | None:
        raise NotImplementedError

    def run(self, partial: Tuple[int, int] | None = None) -> Tensor:
        del partial
        return self.data

    def visit(self, nodes: Dict[int, TensorLike]) -> None:
        raise NotImplementedError

    def device(self) -> str | Device:
        raise NotImplementedError

    def dtype(self) -> DType:
        raise NotImplementedError

    def size(self) -> Tuple[int, ...]:
        return self.data.size()

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Tensor],
        types: Tuple[Type[Any], ...],
        args: Sequence[TensorLike] = (),
        kwargs: Dict[str, TensorLike] | None = None,
    ) -> TensorLike:
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(typ, (Tensor, EagerTensor)) for typ in types):
            return NotImplemented

        return EagerTensor(func(*args, **kwargs))
