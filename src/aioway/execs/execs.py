# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import typing
from abc import ABC
from collections.abc import Callable, Iterable, Iterator
from typing import Self

import structlog

from aioway import registries
from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.nodes import Node
from aioway.procs import OpaqueProc, ProcRewrite

if typing.TYPE_CHECKING:
    from aioway.nodes import Dag

__all__ = ["Exec"]

LOGGER = structlog.get_logger()


@dcls.dataclass(frozen=True)
class ExecCtx(ProcRewrite):
    dag: "Dag"


@dcls.dataclass
class Exec(Iterator[Block], Iterable[Block], Node["Exec"], ABC):
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

    @typing.no_type_check
    def __init_subclass__(cls, *, key: str = ""):
        # Ensure that concrete subclasses can be instantiated from a factory.
        cls.__register_factory_with_key(cls, key=key)

        # Wrap the `__next__` method with the `__proxy_next` method,
        # acting as a decorator but doesn't require adding to every subclasses.
        cls.__next__ = cls.__proxy_next(cls.__next__)

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

        ...

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet:
        """
        The output schema of the current table.
        """

        ...

    @classmethod
    def __proxy_next(
        cls, next_method: Callable[[Self], Block]
    ) -> Callable[[Self], Block]:
        """
        Proxy the next method to the next method of the class.
        """

        if not callable(next_method):
            raise ExecNextMethodError(
                f"`__next__` is not callable, got {type(next_method)}."
            )

        sig = inspect.signature(next_method)

        # Verify the signatures of the `__next__` method.
        if (params := sig.parameters).keys() != {"self"}:
            raise ExecNextMethodError(
                f"`__next__` only takes `self` as argument, got {params} instead."
            )
        if (returns := sig.return_annotation) is not Block:
            raise ExecNextMethodError(
                f"`__next__` is not returning `Block`, got {returns}."
            )

        proc = OpaqueProc(next_method)
        return proc

    __register_factory_with_key = registries.init_subclass(lambda: Exec)
    """
    Register subclasses with the given key into the factory.
    """


class ExecRegisterError(AiowayError, KeyError): ...


class ExecNextMethodError(AiowayError, TypeError): ...
