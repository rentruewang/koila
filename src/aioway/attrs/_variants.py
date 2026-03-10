# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

from aioway._exprs import OpSign

from . import devices, dtypes, shapes
from .attrs import Attr
from .devices import Device, DeviceLike
from .dtypes import DType, DTypeLike
from .sets import AttrSet, DeviceSet, DTypeSet, ShapeSet
from .shapes import Shape, ShapeLike

_UNARY_UFUNC_OPS = "neg", "not"
_ARITH_UFUNC_OPS = "add", "sub", "mul", "truediv", "floordiv", "pow"
_CMP_UFUNC_OPS = "eq", "ne", "gt", "ge", "lt", "le"


@OpSign.ufunc1(Device).register_keys(*_UNARY_UFUNC_OPS)
def same_device(device: DeviceLike) -> Device:
    return devices.device(device)


@OpSign.ufunc2(Device).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
def matching_device(l: DeviceLike, r: DeviceLike) -> Device:
    left, right = map(devices.device, [l, r])

    if left != right:
        raise ValueError(f"{left} != {right}")

    return left


@OpSign.ufunc1(DType).register_keys(*_UNARY_UFUNC_OPS)
def same_dtype(dtype: DTypeLike) -> DType:
    return dtypes.dtype(dtype)


@OpSign.ufunc2(DType).register_keys(*_ARITH_UFUNC_OPS)
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


@OpSign.ufunc2(DType).register_keys(*_CMP_UFUNC_OPS)
def boolean_dtypes(l: DTypeLike, r: DTypeLike) -> DType:
    return dtypes.dtype("bool")


@OpSign.ufunc1(Shape).register_keys(*_UNARY_UFUNC_OPS)
def same_shape(shape: ShapeLike) -> Shape:
    return shapes.shape(shape)


@OpSign.ufunc2(Shape).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
def matching_shape(l: ShapeLike, r: ShapeLike):
    left, right = map(shapes.shape, [l, r])

    if left != right:
        raise ValueError(f"{left} != {right}")

    return left


@dcls.dataclass(frozen=True)
class _UnaryAttrVariant:
    op_name: str

    def __call__(self, attr: Attr) -> Attr:
        return Attr(
            device=OpSign.ufunc1(Device).dispatch(self.op_name)(attr.device),
            dtype=OpSign.ufunc1(DType).dispatch(self.op_name)(attr.dtype),
            shape=OpSign.ufunc1(Shape).dispatch(self.op_name)(attr.shape),
        )


for unary_op in _UNARY_UFUNC_OPS:
    OpSign.ufunc1(Attr).register_keys(unary_op)(_UnaryAttrVariant(unary_op))


@OpSign.ufunc1(AttrSet).register_keys(*_UNARY_UFUNC_OPS)
@OpSign.ufunc1(DeviceSet).register_keys(*_UNARY_UFUNC_OPS)
@OpSign.ufunc1(DTypeSet).register_keys(*_UNARY_UFUNC_OPS)
@OpSign.ufunc1(ShapeSet).register_keys(*_UNARY_UFUNC_OPS)
def same[T](s: T) -> T:
    return s


@OpSign.ufunc2(AttrSet).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
@OpSign.ufunc2(DeviceSet).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
@OpSign.ufunc2(DTypeSet).register_keys(*_ARITH_UFUNC_OPS)
@OpSign.ufunc2(ShapeSet).register_keys(*_ARITH_UFUNC_OPS, *_CMP_UFUNC_OPS)
def matching[T](l: T, r: T) -> T:
    if l != r:
        raise ValueError(f"{l} != {r}")
    return l


@OpSign.ufunc2(DTypeSet).register_keys(*_CMP_UFUNC_OPS)
def boolean(l: DTypeSet, r: DTypeSet) -> DTypeSet:
    if l.keys() != r.keys():
        raise ValueError(f"{list(l.keys())=} != {list(r.keys())=}")

    return DTypeSet.from_dict({k: dtypes.dtype("bool") for k in l.keys()})


@dcls.dataclass(frozen=True)
class _BinaryAttrVariant:
    op_name: str

    def __call__(self, left: Attr, right: Attr) -> Attr:
        device_reg = OpSign.ufunc2(Device).dispatch(self.op_name)
        dtype_reg = OpSign.ufunc2(DType).dispatch(self.op_name)
        shape_reg = OpSign.ufunc2(Shape).dispatch(self.op_name)
        return Attr(
            device=device_reg(left.device, right.device),
            dtype=dtype_reg(left.dtype, right.dtype),
            shape=shape_reg(left.shape, right.shape),
        )


for binary_op in _CMP_UFUNC_OPS + _ARITH_UFUNC_OPS:
    OpSign.ufunc2(Attr).register_keys(binary_op)(_BinaryAttrVariant(binary_op))
