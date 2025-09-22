# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections.abc import Iterable

from aioway import ops
from aioway.ops import BatchGen, Op, Thunk

from .execs import Exec, ExecCtx

__all__ = ["OpExec"]


class OpExec(Exec, key="OP"):
    """
    ``OpExec`` directly accepts ``Op`` as input, as well as the input ``Exec``.
    """

    def __init__(self, op: Op, execs: Iterable[Exec], *ctxs: ExecCtx) -> None:
        super().__init__(*ctxs)
        self._op = op
        self._execs = tuple(execs)

    @typing.override
    def iterate(self) -> BatchGen:
        return self._op(*self._execs)

    @typing.override
    def inputs(self):
        yield from self._execs

    @property
    @typing.override
    def thunk(self) -> Thunk:
        return ops.thunk(self._op, *(e.thunk for e in self._execs))
