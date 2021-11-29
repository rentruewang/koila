import logging
import typing

import torch

from . import *

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]

for logger in loggers:
    logger.setLevel(logging.DEBUG)


def test_buffer_sizes() -> None:
    a = torch.randn(4, 5, 6)

    la = LazyTensor(a)
    assert a.numel() == la.numel() == la.buffer_numel()[1]

    b = torch.randn(4, 5, 1)
    lb = LazyTensor(b)
    assert b.numel() == lb.numel() == lb.buffer_numel()[1]

    lc = typing.cast(LazyTensor, la + lb)
    assert lc.numel() == la.numel() == 6 * lb.numel()
    assert lc.buffer_numel()[1] == la.numel() + lb.numel() + lc.numel()

    d = torch.randn(4, 5, 6)
    ld = typing.cast(LazyTensor, d)

    le = typing.cast(LazyTensor, lc * ld)
    assert d.numel() == ld.numel() == le.numel()
    assert le.buffer_numel()[1] == sum(map(LazyTensor.numel, {la, lb, lc, ld, le}))

    lf = le.sum()
    assert lf.buffer_numel()[1] == sum(map(LazyTensor.numel, {la, lb, lc, ld, le, lf}))

    lg = typing.cast(LazyTensor, lc + le)
    assert lg.buffer_numel()[1] == sum(map(LazyTensor.numel, {la, lb, lc, ld, le, lg}))

    assert lg.buffer_memory()[1] == lg.buffer_numel()[1] * 4


test_buffer_sizes()
