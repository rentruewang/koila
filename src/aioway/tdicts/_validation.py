# Copyright (c) AIoWay Authors - All Rights Reserved

"Validation of data (`td.TensorDict`) against schema (`AttrSet`)."

import numpy as np
import tensordict as td
import torch

from aioway import tensors
from aioway._tracking import logging

from .attrs import AttrSet

__all__ = ["validate_schema", "validate_attr"]

LOGGER = logging.get_logger(__name__)


def validate_schema(attrs: AttrSet, data: td.TensorDict) -> None:
    """
    Validate `data` against `attrs`.

    Currently it loops over all the keys and attributes, which may be slow.

    Args:
        schema: The schema of the tensordict.
        data: The data to validate.

    Raises:
        RuntimeError: If the schema doesn't match the given data.
    """

    LOGGER.debug("Validating data: %s against schema: %s", data, attrs)

    if attrs.keys() != data.keys():
        raise RuntimeError(f"Keys {set(attrs.keys())=} != {set(data.keys())=}")

    for key in attrs.keys():
        validate_attr(attr=attrs[key], tensor=data[key])


import torch


def validate_attr(attr: tensors.Attr, tensor: torch.Tensor) -> None:
    """
    Validate `tensor` against `attr`.

    Only check if `tensor` has the exact same dtype, shape, device as `attr`.
    """

    validate_shape_matches(shape=attr.shape, tensor=tensor)
    validate_device_matches(device=attr.device, tensor=tensor)
    validate_dtype_matches(dtype=attr.dtype, tensor=tensor)


import torch


def validate_shape_matches(shape: tensors.Shape, tensor: torch.Tensor) -> None:
    try:
        _validate_shape_matches(shape, tensor)
    except ValueError:
        raise RuntimeError(
            f"tensors.Shape of tensor {tensor.shape=} should match attr's {shape=}"
        )


import torch


def _validate_shape_matches(shape: tensors.Shape, tensor: torch.Tensor) -> None:
    # Convert to numpy array s.t. we can elegantly formulate the verification.
    left = np.array(shape)
    right = tensor.shape

    # Dimension mismatch.
    if len(left) != len(right):
        raise ValueError

    # If == 1, matches anything, if >= 0, matches `tensor.shape`.
    if np.any((left != right) & (left != 1)):
        raise ValueError


import torch


def validate_dtype_matches(dtype: tensors.DType, tensor: torch.Tensor) -> None:
    if dtype != tensor.dtype:
        raise RuntimeError(
            f"tensors.DType of tensor {tensor.dtype=} should match attr's {dtype=}"
        )


import torch


def validate_device_matches(device: tensors.Device, tensor: torch.Tensor) -> None:
    if device != tensor.device:
        raise RuntimeError(
            f"tensors.Device of tensor {tensor.device=} should match attr's {device=}"
        )
