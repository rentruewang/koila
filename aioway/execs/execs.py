# Copyright (c) RenChu Wang - All Rights Reserved

__all__ = ["Exec"]

import abc
import inspect
from abc import ABC
from collections.abc import Iterable, Iterator

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.factories import Factory
from aioway.plans import PhysicalPlan


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

    @classmethod
    def __init_subclass__(cls, key: str = "") -> None:
        if not key:
            # Allow abstract classes, which would not be initialized,
            # to not define keys, as factories are used to store leaf nodes.
            if inspect.isabstract(cls):
                return

            raise ExecRegisterError(
                f"Class: {cls} isn't given a key argument. Only valid for abstract classes."
            )

        if key in FACTORY:
            raise ExecRegisterError(
                f"Trying to insert key: {key} and class: {cls} "
                f"but key is already used by class: {FACTORY[key]}"
            )

        FACTORY[key] = cls

    @abc.abstractmethod
    def __next__(self) -> Block:
        """
        ``Iterator`` to yield from the current ``Exec``.
        """

        ...

    def __str__(self) -> str:
        """
        todo))
            Use ``reprlib`` or ``pprint`` s.t. we do not rely on ``rich`` in explainer.
        """

        return repr(self)

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet:
        """
        The output schema of the current table.
        """

        ...


FACTORY = Factory(base_class=Exec)
"""
The class factory for ``Exec``s.
"""


class ExecRegisterError(AiowayError, KeyError): ...
