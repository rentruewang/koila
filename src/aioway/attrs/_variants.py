# Copyright (c) AIoWay Authors - All Rights Reserved


from . import devices, shapes
from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .sets import DTypeSet
from .shapes import Shape, ShapeLike

_UNARY_UFUNC_OPS = "neg", "not"
_ARITH_UFUNC_OPS = "add", "sub", "mul", "truediv", "floordiv", "pow"
_CMP_UFUNC_OPS = "eq", "ne", "gt", "ge", "lt", "le"


def same_device(device: DeviceLike) -> Device:
    return Device.parse(device)


def matching_device(l: DeviceLike, r: DeviceLike) -> Device:
    left, right = map(devices.device, [l, r])

    if left != right:
        raise ValueError(f"{left} != {right}")

    return left


def same_dtype(dtype: DTypeLike) -> DType:
    return DType.parse(dtype)


def boolean_dtypes(l: DTypeLike, r: DTypeLike) -> DType:
    return DType.parse("bool")


def same_shape(shape: ShapeLike) -> Shape:
    return Shape.parse(shape)


def matching_shape(l: ShapeLike, r: ShapeLike):
    left, right = map(shapes.shape, [l, r])

    if left != right:
        raise ValueError(f"{left} != {right}")

    return left


def same[T](s: T) -> T:
    return s


def matching[T](l: T, r: T) -> T:
    if l != r:
        raise ValueError(f"{l} != {r}")
    return l


def boolean(l: DTypeSet, r: DTypeSet) -> DTypeSet:
    if l.keys() != r.keys():
        raise ValueError(f"{list(l.keys())=} != {list(r.keys())=}")

    return DTypeSet.from_dict({k: DType.parse("bool") for k in l.keys()})
