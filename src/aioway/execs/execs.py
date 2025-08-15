# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from collections.abc import Iterator
from dataclasses import KW_ONLY as KwOnly
from dataclasses import InitVar
from typing import Any, Self

from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.nodes import Node

if typing.TYPE_CHECKING:
    from .batches import Batch
    from .ops import Op
    from .pollers import Poller

__all__ = ["Exec"]

LOGGER = logging.getLogger(__name__)


@typing.final
@dcls.dataclass(frozen=False)
class Exec(Node["Exec"]):
    """
    ``Exec`` is the graph / symbolic representation of an execution.

    An execution itself does not store state,
    but rather launches an iterator / cursor to iterate over the data.
    This design allows users to write genertors (``__iter__`` funciton),
    rather than iterators (``__next__`` function) with state management.
    """

    poller: "Poller[Batch]" = dcls.field(init=False)
    """
    The ``Poller`` that determines the iteration strategy,
    as well as storing the children of the current ``Exec``.
    """

    op: "Op" = dcls.field(init=False)
    """
    The ``Op`` to apply on retrieved data from the poller.
    It shall be a pure function.
    """

    _: KwOnly

    poller_type: InitVar[type["Poller"]]
    """
    The type of ``Poller``.
    """

    execs: InitVar[tuple[Self, ...]]
    """
    The children for the current ``Exec``, passed to ``poller_type`` to construct a ``Poller``.
    """

    op_type: InitVar[type["Op"]]
    """
    The type of ``Op``.
    """

    op_opts: InitVar[dict[str, Any]]
    """
    The options / kwargs for ``Op``.
    """

    def __post_init__(
        self,
        poller_type: type["Poller"],
        execs: tuple[Self, ...],
        op_type: type["Op"],
        op_opts: dict[str, Any],
    ) -> None:
        if poller_type.ARGC != op_type.ARGC:
            raise ExecInitError(f"{poller_type.ARGC=} != {op_type.ARGC=}.")

        self.poller = poller_type.init(*execs)
        self.op = op_type(**op_opts)

    def __iter__(self) -> Iterator[Block]:
        for batch in self.poller:
            yield self.op(batch)

    @property
    @typing.override
    def children(self) -> tuple[Self, ...]:
        return self.poller.execs


class ExecInitError(AiowayError, TypeError): ...
