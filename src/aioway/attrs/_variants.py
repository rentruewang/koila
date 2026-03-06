# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway import variants
from aioway.variants import ParamList

from . import devices, dtypes, shapes
from .attrs import Attr
from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .sets import AttrSet, DeviceSet, DTypeSet, ShapeSet
from .shapes import Shape, ShapeLike

_UNARY_UFUNC_OPS = "neg", "not"
_ARITH_UFUNC_OPS = "add", "sub", "mul", "truediv", "floordiv", "pow"
_CMP_UFUNC_OPS = "eq", "ne", "gt", "ge", "lt", "le"


@variants.register(ParamList(Device), *_UNARY_UFUNC_OPS)
def same_device(device: DeviceLike) -> Device:
    return devices.device(device)


@variants.register(ParamList(Device, Device), *_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
def matching_device(l: DeviceLike, r: DeviceLike) -> Device:
    left, right = map(devices.device, [l, r])

    if left != right:
        raise ValueError(f"{left} != {right}")

    return left


@variants.register(ParamList(DType), *_UNARY_UFUNC_OPS)
def same_dtype(dtype: DTypeLike) -> DType:
    return dtypes.dtype(dtype)


@variants.register(ParamList(DType, DType), *_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
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


@variants.register(ParamList(Shape), *_UNARY_UFUNC_OPS)
def same_shape(shape: ShapeLike) -> Shape:
    return shapes.shape(shape)


@variants.register(ParamList(Shape, Shape), *_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
def matching_shape(l: ShapeLike, r: ShapeLike):
    left, right = map(shapes.shape, [l, r])

    if left != right:
        raise ValueError(f"{left} != {right}")

    return left


@variants.register(ParamList(Attr), *_UNARY_UFUNC_OPS)
@variants.register(ParamList(AttrSet), *_UNARY_UFUNC_OPS)
@variants.register(ParamList(DeviceSet), *_UNARY_UFUNC_OPS)
@variants.register(ParamList(DTypeSet), *_UNARY_UFUNC_OPS)
@variants.register(ParamList(ShapeSet), *_UNARY_UFUNC_OPS)
def same[T](s: T) -> T:
    return s


@variants.register(ParamList(Attr, Attr))
@variants.register(ParamList(AttrSet, AttrSet), *_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
@variants.register(ParamList(DeviceSet, DeviceSet), *_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
@variants.register(ParamList(DTypeSet, DTypeSet), *_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
@variants.register(ParamList(ShapeSet, ShapeSet), *_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
def matching[T](l: T, r: T) -> T:
    if l != r:
        raise ValueError(f"{l} != {r}")
    return l
