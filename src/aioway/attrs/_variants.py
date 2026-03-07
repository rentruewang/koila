# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.variants import Signature

from . import devices, dtypes, shapes
from .attrs import Attr
from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .sets import AttrSet, DeviceSet, DTypeSet, ShapeSet
from .shapes import Shape, ShapeLike

_UNARY_UFUNC_OPS = "neg", "not"
_ARITH_UFUNC_OPS = "add", "sub", "mul", "truediv", "floordiv", "pow"
_CMP_UFUNC_OPS = "eq", "ne", "gt", "ge", "lt", "le"


@Signature.ufunc1(Device).register_keys(*_UNARY_UFUNC_OPS)
def same_device(device: DeviceLike) -> Device:
    return devices.device(device)


@Signature.ufunc2(Device).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
def matching_device(l: DeviceLike, r: DeviceLike) -> Device:
    left, right = map(devices.device, [l, r])

    if left != right:
        raise ValueError(f"{left} != {right}")

    return left


@Signature.ufunc1(DType).register_keys(*_UNARY_UFUNC_OPS)
def same_dtype(dtype: DTypeLike) -> DType:
    return dtypes.dtype(dtype)


@Signature.ufunc2(DType).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
def binary_broadcast(l: DTypeLike, r: DTypeLike) -> DType:
    left, right = map(dtypes.dtype, [l, r])

    larger = max(left.bits, right.bits)
    match left.family, right.family:
        case ("float", _) | (_, "float"):
            return dtypes.dtype(f"float{larger}")
        case ("int", _) | (_, "int"):
            return dtypes.dtype(f"int{larger}")
        case _:
            return dtypes.dtype("bool")


@Signature.ufunc1(Shape).register_keys(*_UNARY_UFUNC_OPS)
def same_shape(shape: ShapeLike) -> Shape:
    return shapes.shape(shape)


@Signature.ufunc2(Shape).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
def matching_shape(l: ShapeLike, r: ShapeLike):
    left, right = map(shapes.shape, [l, r])

    if left != right:
        raise ValueError(f"{left} != {right}")

    return left


@Signature.ufunc1(Attr).register_keys(*_UNARY_UFUNC_OPS)
@Signature.ufunc1(AttrSet).register_keys(*_UNARY_UFUNC_OPS)
@Signature.ufunc1(DeviceSet).register_keys(*_UNARY_UFUNC_OPS)
@Signature.ufunc1(DTypeSet).register_keys(*_UNARY_UFUNC_OPS)
@Signature.ufunc1(ShapeSet).register_keys(*_UNARY_UFUNC_OPS)
def same[T](s: T) -> T:
    return s


@Signature.ufunc2(Attr).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
@Signature.ufunc2(AttrSet).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
@Signature.ufunc2(DeviceSet).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
@Signature.ufunc2(DTypeSet).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
@Signature.ufunc2(ShapeSet).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
def matching[T](l: T, r: T) -> T:
    if l != r:
        raise ValueError(f"{l} != {r}")
    return l
