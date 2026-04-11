# Copyright (c) AIoWay Authors - All Rights Reserved

"Validation of data (`td.TensorDict`) against schema (`meta.AttrSet`)."

import logging

import tensordict as td
import torch

from aioway.schemas import Attr, AttrSet, Device, DType, Shape

__all__ = ["validate_schema", "validate_attr"]

LOGGER = logging.getLogger(__name__)


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


def validate_attr(attr: Attr, tensor: torch.Tensor) -> None:
    """
    Validate `tensor` against `attr`.

    Only check if `tensor` has the exact same dtype, shape, device as `attr`.
    """

    validate_shape_larger(max_shape=attr.shape, tensor=tensor)
    validate_device_matches(device=attr.device, tensor=tensor)
    validate_dtype_matches(dtype=attr.dtype, tensor=tensor)


def validate_shape_larger(max_shape: Shape, tensor: torch.Tensor) -> None:

    try:
        _validate_shape_larger(max_shape, tensor)
    except ValueError:
        raise RuntimeError(
            f"attrs.Shape of tensor {tensor.shape=} should match attr's {max_shape=}"
        )


def _validate_shape_larger(max_shape: Shape, tensor: torch.Tensor) -> None:
    # Convert to numpy array s.t. we can elegantly formulate the verification.
    tensor_shape = Shape.parse(tensor.shape)

    if tensor_shape.exceeds(max_shape):
        raise ValueError


def validate_dtype_matches(dtype: DType, tensor: torch.Tensor) -> None:
    if dtype != tensor.dtype:
        raise RuntimeError(
            f"meta.DType of tensor {tensor.dtype=} should match attr's {dtype=}"
        )


def validate_device_matches(device: Device, tensor: torch.Tensor) -> None:
    if device != tensor.device:
        raise RuntimeError(
            f"meta.Device of tensor {tensor.device=} should match attr's {device=}"
        )
