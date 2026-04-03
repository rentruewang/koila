# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import torch

from aioway import _common

from . import tensors

__all__ = ["TensorDataFn"]


@_common.dcls_no_eq
class TensorDataFn(tensors.BasicPreviewFn):
    "The `fn.Fn` representing a plain `torch.Tensor`."

    data: torch.Tensor

    @typing.override
    def do(self) -> torch.Tensor:
        return self.data

    @typing.override
    def deps(self):
        return ()
