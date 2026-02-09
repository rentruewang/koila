# Copyright (c) AIoWay Authors - All Rights Reserved

import logging

from numpy import ndarray as NpArr
from torch import Tensor

from .sets import AttrSet

LOGGER = logging.getLogger(__name__)


def renames(schema: AttrSet, **renames: str) -> AttrSet:
    return AttrSet.from_dict(
        {renames.get(key, key): attr for key, attr in schema.items()}
    )


def index(schema: AttrSet, idx: int | slice | list[int] | NpArr | Tensor) -> AttrSet:
    names = schema.names
    devices = schema.devices
    shapes = schema.shapes
    dtypes = schema.dtypes

    if isinstance(idx, int):
        modified = [shape[1:] for shape in shapes]

    elif isinstance(idx, slice | NpArr | Tensor):
        modified = shapes[:]

    elif isinstance(idx, list) and all(isinstance(i, int) for i in idx):
        modified = shapes[:]

    else:
        raise TypeError(f"{type(idx)=} is not supported.")

    return AttrSet.from_fields(
        names=names, devices=devices, dtypes=dtypes, shapes=modified
    )
