# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import typing

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.execs.execs import Exec

from .unary import UnaryExec

__all__ = ["ProjectExec", "RenameExec", "EchoExec"]


@typing.final
@dcls.dataclass
class ProjectExec(UnaryExec, key="PROJECT"):
    """
    Select a subset of the columns.
    """

    subset: list[str] | None = None
    """
    The subset to use. If not give, the ``ProjectStream`` would be a null operation.
    """

    def __post_init__(self) -> None:
        subs = self.subset

        if subs is None:
            return

        if not isinstance(subs, list) and all(isinstance(c, str) for c in subs):
            raise ProjectColumnTypeError("Column must be a list of strings.")

    @typing.override
    def __next__(self) -> Block:
        item = next(self.child)
        return item if self.subset is None else item[self.subset]

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        schema = self.child.attrs

        if self.subset is None:
            return schema

        return schema.project(*self.subset)


@typing.final
@dcls.dataclass(init=False)
class RenameExec(UnaryExec, key="RENAME"):
    """
    Rename a couple of columns.
    """

    renames: dict[str, str] = dcls.field(default_factory=dict)
    """
    The mapping dictionary names.
    """

    def __init__(self, __exe: Exec, /, **renames: str) -> None:
        # This constructor is provideds.t. `RenameStream`'s renames can be specified as **kwargs,
        # which means they will be variable names, consistent with what `TensorDict` provides.
        #
        # Even though I'm using a Python version with positional only argument,
        # since ``Exec`` is common, using ``__exe`` to avoid name collision (in keys).
        self.child = __exe
        self.renames = renames

    @typing.override
    def __next__(self) -> Block:
        item = next(self.child)
        return item.rename(**self.renames)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.child.attrs.rename(**self.renames)


@typing.final
@dcls.dataclass
class EchoExec(UnaryExec, key="ECHO"):
    """
    ``EchoExec`` repeats what the input says for the specified number of times.
    """

    times: int

    @typing.override
    def __next__(self) -> Block:
        return next(self._generator)

    @functools.cached_property
    def _generator(self):
        for block in self.child:
            for _ in range(self.times):
                yield block

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.child.attrs


class ProjectColumnTypeError(AiowayError, TypeError): ...
