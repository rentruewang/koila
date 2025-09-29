# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch._subclasses import FakeTensorMode

from aioway._attrs.dtypes import DType, StringDType, TorchDType


def promote_dtype(*dtypes: str) -> DType:
    """
    Using ``FakeTensorMode`` from ``torch``,
    get the tuple of shape, dtype.
    """

    with FakeTensorMode():
        tensors = [
            torch.zeros(
                size=[1], dtype=TorchDType.create(StringDType(dt).parse()).dtype
            )
            for dt in dtypes
        ]
        result = torch.cat(tensors)

    return TorchDType(result.dtype)


def dtype_word_size(dtype: str):
    match dtype:
        case "int" | "int64" | "float" | "float64":
            return 64
        case "int32" | "float32":
            return 32
        case "int16" | "float16":
            return 16
        case "int8" | "char" | "bool":
            return 8
        case _:
            raise NotImplementedError
