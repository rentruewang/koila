# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import typing
from abc import ABC
from collections.abc import Iterable, Iterator

from aioway import factories
from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.plans import PhysicalPlan

__all__ = ["Exec"]


class Exec(Iterator[Block], Iterable[Block], PhysicalPlan, ABC):
    """
    ``Exec`` represents a stream of heterogenious data being generated,
    it is one of the main physical abstractions in ``aioway`` to represent eager computation.

    ``Exec`` acts like ``torch``'s ``DataLoader``'s iterator,
    where it is an ``Iterator`` and ``Iterable``,
    so it can be used in for loops and yield expressions easily.

    It can be thought of as an ``Iterator`` of ``Block``s,
    where computation happens eagerly, imperatively, and the result is yielded.

    This design decision is made because we would like to enable lazy / iterator processing,
    and if we directly follow the abstraction of ``IterableDataset``,
    we have to process the tensor representation of the items 1 by 1, which can be inefficient.
    """

    # NOTE Keep until the issue python/mypy#18987 is fixed.
    if typing.TYPE_CHECKING:

        def __init_subclass__(cls, *, key: str = ""): ...

    else:
        __init_subclass__ = factories.init_subclass(lambda: Exec)

    def __hash__(self) -> int:
        """
        The unique identifier of each node, representing computation.

        This means that ``__hash__`` would be the same for shared computation.

        For now, it is the ``id`` of the object itself,
        and the object itself is the unique identifier of the computation.
        However, in the future, we might want to use a more sophisticated way to identify computation
        for distributed execution and caching.
        """

        return id(self)

    @abc.abstractmethod
    def __next__(self) -> Block:
        """
        ``Iterator`` to yield from the current ``Exec``.
        """

        ...

    def __str__(self) -> str:
        # TODO Use `reprlib` or `pprint` s.t. we do not rely on `rich` in explainer.
        return repr(self)

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet:
        """
        The output schema of the current table.
        """

        ...


class ExecRegisterError(AiowayError, KeyError): ...
