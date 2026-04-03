# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import torch

from aioway import _common, fake

from . import tensors

__all__ = ["TensorDataFn"]


@_common.dcls_no_eq
class TensorDataFn(tensors.TensorFn):
    "The `fn.Fn` representing a plain `torch.Tensor`."

    data: torch.Tensor

    def __post_init__(self) -> None:
        super().__init__()

        # Mark as `EVALUATED`.
        _ = self.do()

    @typing.override
    def do(self):

        data = self.data

        if fake.is_enabled():
            data = fake.to_fake_tensor(data)

        return data

    @typing.override
    def deps(self):
        return ()
