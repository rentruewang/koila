# Copyright (c) AIoWay Authors - All Rights Reserved

"Some additional functions on `TensorDict`s."

import torch
from tensordict import TensorDict
from torch import Tensor

from aioway import _logging

LOGGER = _logging.get_logger(__name__)


def to_tensor(td: TensorDict) -> Tensor:
    """
    Convert a `TensorDict` to a `Tensor`, by treating the dict as a table.
    Respects the `.values()` orders of `TensorDict`s.
    """

    columns: list[Tensor] = []

    for value in td.values():
        columns.append(value.view(len(value), -1))

    return torch.cat(columns, dim=1)
