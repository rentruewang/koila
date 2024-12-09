# Copyright (c) RenChu Wang - All Rights Reserved

import functools

import torch
from torch import Tensor

from aioway.schemas import DataType, DataTypeEnum


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


def batched_tensor_to_aioway_dtype(tensor: Tensor) -> DataType:
    if tensor.ndim == 0:
        raise ValueError("Tensor must be batched, therefore cannot be 0D.")

    dtype = _dtype_dict()[tensor.dtype]

    if tensor.ndim == 1:
        return dtype

    return DataTypeEnum.ARRAY(tensor.shape[1:], dtype)
