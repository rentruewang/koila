from koila import Function, Runnable, Tensor
import torch

import koila


def test_func() -> None:
    a = torch.tensor(1)
    b = torch.tensor(-1)
    func = Function(torch.add, (a, b), {})
    assert func.run().item() == 0


def test_add() -> None:
    a = koila.lazy(torch.tensor(1))
    b = koila.lazy(torch.tensor(2))
    c = a + b
    assert c.run().item() == 3
