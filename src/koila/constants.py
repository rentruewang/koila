from typing import Dict

import torch
from torch import dtype

UNITS: Dict[str, int] = {
    "b": 1,
    "kb": 10 ** 3,
    "kib": 2 ** 10,
    "mb": 10 ** 6,
    "mib": 2 ** 20,
    "gb": 10 ** 9,
    "gib": 2 ** 30,
    "tb": 10 ** 4,
    "tib": 2 ** 40,
}

MEMORY_BYTES: Dict[dtype, int] = {
    torch.bool: 1,
    torch.uint8: 1,
    torch.int8: 1,
    torch.short: 2,
    torch.int16: 2,
    torch.int: 4,
    torch.int32: 4,
    torch.long: 8,
    torch.int64: 8,
    torch.half: 2,
    torch.float16: 2,
    torch.float: 4,
    torch.float32: 4,
    torch.double: 8,
    torch.float64: 8,
}
