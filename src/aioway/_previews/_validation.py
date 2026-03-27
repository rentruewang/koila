# Copyright (c) AIoWay Authors - All Rights Reserved

"Validation of data (`TensorDict`) against schema (`AttrSet`)."

import numpy as np
from tensordict import TensorDict
from torch import Tensor

from aioway._tracking import logging

from .attrs import Attr
from .devices import Device
from .dtypes import DType
from .sets import AttrSet
from .shapes import Shape

__all__ = ["validate_schema", "validate_attr"]

LOGGER = logging.get_logger(__name__)


def validate_schema(attrs: AttrSet, data: TensorDict) -> None:
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


def validate_attr(attr: Attr, tensor: Tensor) -> None:
    """
    Validate `tensor` against `attr`.

    Only check if `tensor` has the exact same dtype, shape, device as `attr`.
    """

    validate_shape_matches(shape=attr.shape, tensor=tensor)
    validate_device_matches(device=attr.device, tensor=tensor)
    validate_dtype_matches(dtype=attr.dtype, tensor=tensor)


def validate_shape_matches(shape: Shape, tensor: Tensor) -> None:
    try:
        _validate_shape_matches(shape, tensor)
    except ValueError:
        raise RuntimeError(
            f"Shape of tensor {tensor.shape=} should match attr's {shape=}"
        )


def _validate_shape_matches(shape: Shape, tensor: Tensor) -> None:
    # Convert to numpy array s.t. we can elegantly formulate the verification.
    left = np.array(shape)
    right = tensor.shape

    # Dimension mismatch.
    if len(left) != len(right):
        raise ValueError

    # If == 1, matches anything, if >= 0, matches `tensor.shape`.
    if np.any((left != right) & (left != 1)):
        raise ValueError


def validate_dtype_matches(dtype: DType, tensor: Tensor) -> None:
    if dtype != tensor.dtype:
        raise RuntimeError(
            f"DType of tensor {tensor.dtype=} should match attr's {dtype=}"
        )


def validate_device_matches(device: Device, tensor: Tensor) -> None:
    if device != tensor.device:
        raise RuntimeError(
            f"Device of tensor {tensor.device=} should match attr's {device=}"
        )
