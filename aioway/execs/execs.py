# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import logging
from abc import ABC
from collections.abc import Callable, Iterable, Iterator
from typing import Protocol

from aioway import factories
from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.plans import PlanNode

__all__ = ["Exec"]

LOGGER = logging.getLogger(__name__)


class NextMethod(Protocol):
    def __call__(self) -> Block: ...

    def __get__(self, instance: "Exec", owner: type["Exec"]) -> Callable[[], Block]: ...


class Exec(Iterator[Block], Iterable[Block], PlanNode["Exec"], ABC):
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

    def __init_subclass__(cls, *, key: str = ""):
        init_factory_key = factories.init_subclass(lambda: Exec)
        init_factory_key(cls, key=key)

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

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self) -> Block:
        """
        Compute the next block of data.

        Note:
            Since ``Exec``s are a poll system of ``Iterator``s, those iterators must be in sync.
            However, having a DAG of ``Exec``s means that some of the computation is shared,
            which is extremely common place.

            However, if we follow a naive ``__next__`` scheme, this means that
            not all of the ``Exec``s are only called once, which can lead to bugs.

            Assuming that ``Exec``s are public API, we must choose from one of the cases:

            #. The computation is not shared,
                and the same computation is performed multiple times.
            #. The computation is shared, by means of some buffer.
                This needs to be special cased and is not scalable.
            #. The computation is shared with a new ``Exec``, say ``BufferExec``.
                However, this ``BufferExec`` is not a good abstraction,
                because calling ``next`` 2 times should give the same result
                until buffer is cleared, which violates basic assumption of `next`.

            None of which is worth it.

            However, if ``Exec``s are not public API, we can manage the ``Iterator``s ourselves,
            which means that we can make use of the following patterns:

            #. Call ``next`` carefully.
            #. Use contexts to manage the execution.
            #. Use decorators to manage the execution.
        """

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet:
        """
        The output schema of the current table.
        """

        ...


class ExecRegisterError(AiowayError, KeyError): ...
