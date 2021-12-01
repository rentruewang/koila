from __future__ import annotations

import koila
from koila import Evaluation, LazyTensor, Runnable, RunnableTensor


def test_is_runnable() -> None:
    class MyCustomRunnable:
        num_elements = 0

        def run(self, partial=None) -> int:
            return 42

    mcr = MyCustomRunnable()
    assert isinstance(mcr, Runnable)

    assert koila.run(mcr) == 42


def test_lazytensor_is_runnable() -> None:
    assert issubclass(Evaluation, Runnable)
    assert issubclass(Evaluation, RunnableTensor)
    assert issubclass(LazyTensor, Runnable)
    assert issubclass(LazyTensor, RunnableTensor)
