from __future__ import annotations

from typing import Mapping, Sequence

from torch import Tensor

from koila.interfaces import RunnableTensor, TensorLike

# class DelayedTensor(TensorLike):
#     def __init__(self) -> None:
#         pass


class DelayedComputation(RunnableTensor, TensorLike):
    def __init__(
        self,
        *,
        function: callable[[TensorLike, ...], TensorLike],
        args: Sequence[TensorLike],
        kwargs: Mapping[str, TensorLike],
    ) -> None:
        super().__init__()
        self._function = function
        self._cached_args = args
        self._cached_kwargs = kwargs

    def run(self, partial: range | None = None) -> Tensor:
        assert partial is None, partial
        return self._function(*self._cached_args, **self._cached_kwargs)
