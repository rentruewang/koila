# Copyright (c) AIoWay Authors - All Rights Reserved

"``Vector`` is a homogenious array of values."

import dataclasses as dcls
import operator
import typing
from collections.abc import Callable

from torch import Tensor

from aioway import attrs, variants
from aioway.attrs import Attr, TorchDType, _validation
from aioway.variants import Signature

__all__ = ["Vector"]


class Vector:
    """
    A ``Vector`` is a ``Tensor`` plus its ``Attr``.
    """

    def __init__(self, data: Tensor, attr: Attr) -> None:
        # Validate the attribute.
        _validation.validate_attr(attr=attr, tensor=data)

        self.attr = attr
        "The attribute that the ``Tensor`` must satisfy."

        self.data = data
        "The underlying data."

    def __repr__(self):
        return f"Vector(data={self.data}, attr={self.attr})"

    def __add__(self, rhs):
        return variants.find(Signature.ufunc2(Vector), "add")(self, rhs)

    def __sub__(self, rhs):
        return variants.find(Signature.ufunc2(Vector), "sub")(self, rhs)

    def __mul__(self, rhs):
        return variants.find(Signature.ufunc2(Vector), "mul")(self, rhs)

    def __truediv__(self, rhs):
        return variants.find(Signature.ufunc2(Vector), "truediv")(self, rhs)

    def __floordiv__(self, rhs):
        return variants.find(Signature.ufunc2(Vector), "floordiv")(self, rhs)

    def __pow__(self, rhs):
        return variants.find(Signature.ufunc2(Vector), "pow")(self, rhs)

    def __eq__(self, rhs):
        return self._cmp_op(rhs, operator.eq)

    def __ne__(self, rhs):
        return self._cmp_op(rhs, operator.ne)

    def __le__(self, rhs):
        return self._cmp_op(rhs, operator.le)

    def __lt__(self, rhs):
        return self._cmp_op(rhs, operator.lt)

    def __ge__(self, rhs):
        return self._cmp_op(rhs, operator.ge)

    def __gt__(self, rhs):
        return self._cmp_op(rhs, operator.gt)

    def _cmp_op(self, rhs, op: Callable):
        data = op(self.data, rhs)
        attr = attr = attrs.attr(
            device=self.attr.device,
            shape=self.attr.shape,
            dtype="bool",
        )
        return type(self)(data=data, attr=attr)

    def tolist(self):
        return self.data.tolist()

    def torch(self) -> Tensor:
        return self.data

    def cpu(self):
        data = self.data.cpu()
        attr = attrs.attr(
            device="cpu",
            shape=self.attr.shape,
            dtype=self.attr.dtype,
        )
        return type(self)(data=data, attr=attr)

    def numpy(self):
        return self.torch().cpu().numpy()


@dcls.dataclass(frozen=True)
class VectorBinary:

    op: Callable

    @typing.no_type_check
    def __call__(
        self, lhs: Vector, rhs: Vector | Tensor | int | float | bool
    ) -> Vector:
        match rhs:
            case Vector():
                rhs_data = rhs.data
                rhs_attr = rhs.attr
            case Tensor():
                rhs_data = rhs
                rhs_attr = attrs.attr(
                    device=rhs.device,
                    dtype=TorchDType(rhs.dtype),
                    shape=rhs.shape,
                )
            case _:
                rhs_data = rhs
                rhs_attr = attrs.attr(
                    device="cpu", dtype=rhs.__class__.__name__, shape=[]
                )

        data = self.op(lhs.data, rhs_data)
        attr_func = variants.find(Signature(Attr, type(rhs), Attr), self.op.__name__)
        attr = attr_func(lhs.attr, rhs_attr)
        return Vector(data=data, attr=attr)


for op in [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.eq,
    operator.ne,
    operator.le,
    operator.lt,
    operator.ge,
    operator.gt,
]:
    comparator = VectorBinary(op)

    Signature(Vector, Vector, Vector).register_keys(op.__name__)(comparator)
    Signature(Vector, Tensor, Vector).register_keys(op.__name__)(comparator)
    Signature(Vector, int, Vector).register_keys(op.__name__)(comparator)
    Signature(Vector, float, Vector).register_keys(op.__name__)(comparator)
    Signature(Vector, bool, Vector).register_keys(op.__name__)(comparator)
