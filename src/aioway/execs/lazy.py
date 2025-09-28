# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway import thunks
from aioway._errors import AiowayError
from aioway.ops import BatchGen, Op
from aioway.thunks import Thunk

from .execs import Exec

__all__ = ["OpExec", "TreeExec"]


class OpExec(Exec):
    """
    ``OpExec`` directly accepts ``Op`` as input, as well as the input ``Exec``.
    """

    def __init__(self, op: Op, execs: tuple[Exec, ...]):
        super().__init__()
        self._op = op
        self._execs = execs

        if len(self.execs) != self.op.ARGC:
            raise OpOperandError(
                f"The number of operands {len(self.execs)} does not match "
                f"{self.op=}'s requirement {self.op.ARGC}."
            )

    @typing.override
    def __repr__(self) -> str:
        args = ", ".join(map(str, (self.op, *self.execs)))
        return f"OpExec({args})"

    @typing.override
    def iter(self) -> BatchGen:
        return self.op.apply(*self.inputs())

    @typing.override
    def inputs(self):
        yield from self.execs

    @property
    @typing.override
    def thunk(self) -> Thunk:
        return thunks.thunk(self.op, *(e.thunk for e in self.execs))

    @property
    def op(self) -> Op:
        """
        The operator for which we should use to apply.
        """

        return self._op

    @property
    def execs(self) -> tuple[Exec, ...]:
        """
        The executors operands to the ``Op``'s inputs.
        """
        return self._execs


class TreeExec(Exec):
    """
    ``TreeExec`` is the simplest implementation of ``Exec``,
    iterates over the graph with lazy evaluation.
    """

    def __init__(self, thunk: Thunk, /) -> None:
        super().__init__()
        self._thunk: Thunk = thunk

    @typing.override
    def __repr__(self) -> str:
        args = ", ".join(map(str, self.thunk.inputs()))
        return f"TreeExec({args})"

    @typing.override
    def iter(self) -> BatchGen:
        """
        Yields a generator, locally, from ``Op``'s definition.

        Returns:
            A stream of ``Block``s.

        Note:
            Always creates a new ``Generator`` upon being called, not cached.
        """

        yield from self._thunk.op.apply(*self.inputs())

    @typing.override
    def inputs(self):
        """
        ``TreeExec.input()`` recursively wraps their inputs as ``TreeExec``s.
        """
        for th in self._thunk.inputs():
            yield TreeExec(th)

    @property
    @typing.override
    def thunk(self):
        return self._thunk


class OpOperandError(AiowayError, TypeError): ...
