# Copyright (c) RenChu Wang - All Rights Reserved

import functools

import torch
from torch import Tensor

from aioway.schemas import ArrayDtype, DataTypeEnum


@functools.cache
def _dtype_dict():
    return {
        torch.bool: DataTypeEnum.BOOL(),
        torch.int16: DataTypeEnum.INT(precision=16),
        torch.int32: DataTypeEnum.INT(precision=32),
        torch.int64: DataTypeEnum.INT(precision=64),
        torch.float16: DataTypeEnum.FLOAT(precision=16),
        torch.float32: DataTypeEnum.FLOAT(precision=32),
        torch.float64: DataTypeEnum.FLOAT(precision=64),
    }


def tensor_dtype(tensor: Tensor) -> ArrayDtype:
    if tensor.ndim == 0:
        raise ValueError("Tensor must be batched, therefore cannot be 0D.")

    dtype = _dtype_dict()[tensor.dtype]

    return ArrayDtype(shape=tensor.shape, dtype=dtype)
