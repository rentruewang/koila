# Copyright (c) AIoWay Authors - All Rights Reserved

"Some additional functions on `TensorDict`s."

import torch
from tensordict import TensorDict

from aioway._tracking import logging

LOGGER = logging.get_logger(__name__)


def to_tensor(td: TensorDict) -> torch.Tensor:
    """
    Convert a `TensorDict` to a `torch.Tensor`, by treating the dict as a table.
    Respects the `.values()` orders of `TensorDict`s.
    """

    columns: list[torch.Tensor] = []

    for value in td.values():
        columns.append(value.view(len(value), -1))

    return torch.cat(columns, dim=1)
