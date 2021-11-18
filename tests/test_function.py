import torch

import koila
from koila import LazyFunction
from koila.protocols import Runnable


def test_func_is_runnable() -> None:
    assert issubclass(LazyFunction, Runnable)


def test_add() -> None:
    a = koila.lazy(torch.tensor(1))
    b = koila.lazy(torch.tensor(2))
    c = a + b
    d = a.add(b)
    assert c.run().item() == 3
    assert d.run().item() == c.run().item()
