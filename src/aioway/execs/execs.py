# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from abc import ABC
from collections.abc import Iterable

from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.ops import BlockGen, BlockIter, Op

__all__ = ["Exec"]

LOGGER = logging.getLogger(__name__)


@typing.final
@dcls.dataclass(init=False)
class Exec(Iterable[Block], ABC):
    """
    ``Exec`` is the graph / symbolic representation of an execution.

    An execution itself does not store state,
    but rather launches an iterator / cursor to iterate over the data.
    This design allows users to write genertors (``__iter__`` funciton),
    rather than iterators (``__next__`` function) with state management.
    """

    op: Op
    """
    The operator for which to execute.
    """

    inputs: tuple[BlockIter, ...]
    """
    The recursive dependency on previous executors.
    """

    def __init__(self, op: Op, *inputs: BlockIter) -> None:
        self.op = op
        self.inputs = inputs

    def __post_init__(self) -> None:
        if self.argc != self.op.ARGC:
            raise ExecArgcError(
                f"Operator {self.op} only takes {self.op.ARGC} inputs. Got {self.argc} inputs: {self.inputs}."
            )

    def __iter__(self) -> BlockGen:
        """
        The ``__iter__`` method launches a new ``Iterator`` to loop over the inputs.
        Every call is creates / rebuilds brand new computation.

        Returns:
            A stream of ``Block``s.

        Note:
            Currently ``Exec`` always creates a new ``Generator`` upon being called.
            Perhaps implement STG (#77).
        """

        yield from self.op.apply(*self.inputs)

    @property
    def argc(self) -> int:
        "Argument count for ``Exec``."

        return len(self.inputs)


class ExecArgcError(AiowayError, TypeError): ...
