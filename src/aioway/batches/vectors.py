# Copyright (c) AIoWay Authors - All Rights Reserved

"``Vector`` is a homogenious array of values."

import dataclasses as dcls
import operator
from dataclasses import InitVar

from torch import Tensor

from aioway import attrs
from aioway.attrs import Attr, _validation
from aioway.tables import Column

__all__ = ["Vector"]


@dcls.dataclass(frozen=True)
class Vector(Column):
    """
    A ``Vector`` is a ``Tensor`` plus its ``Attr``.
    """

    data: Tensor
    "The underlying data."

    attr: Attr
    "The attribute thta the ``Tensor`` must satisfy."

    validate: InitVar[bool] = True
    "Whether or not to perform validation (exists for performance)."

    def __post_init__(self, validate: bool) -> None:
        # Validate the attribute.
        if validate:
            _validation.validate_attr(attr=self.attr, tensor=self.data)

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

    def _cmp_op(self, rhs, op):
        data = op(self.data, rhs)
        attr = attrs.attr(
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
