import torch

from koila import EagerTensor

from . import common


def test_add_function() -> None:
    a = EagerTensor(torch.tensor([1, 2, 3]))
    b = EagerTensor(torch.tensor([4, 3, 2]))
    common.call(lambda a, b, c: common.assert_equal, [[a, b, torch.tensor([5, 5, 5])]])
