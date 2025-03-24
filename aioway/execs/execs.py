# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import inspect
from abc import ABC
from collections.abc import Iterable, Iterator

from aioway.blocks import Block
from aioway.datatypes import AttrSet
from aioway.errors import AiowayError

__all__ = ["Exec"]


class Exec(Iterator[Block], Iterable[Block], ABC):
    """
    ``Exec`` represents a stream of heterogenious data being generated,
    it is one of the main physical abstractions in ``aioway`` to represent eager computation.

    ``Exec`` acts like a generator, where it is an ``Iterator`` and ``Iterable``,
    so it can be used in for loops and yield expressions easily.

    It can be thought of as an ``Iterator`` of ``Block``s,
    where computation happens eagerly, imperatively, and the result is yielded.
    """

    FACTORY: dict[str, type["Exec"]] = {}
    """
    The class factory for ``Exec``s.
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

        if exists := cls.FACTORY.get(key):
            raise ExecRegisterError(
                f"Trying to insert key: {key} and class: {cls} "
                f"but key is already used by class: {exists}"
            )

        cls.FACTORY[key] = cls

    @abc.abstractmethod
    def __next__(self) -> Block:
        """
        ``Iterator`` to yield from the current ``Exec``.
        """

        ...

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet:
        """
        The output schema of the current table.
        """

        ...


class ExecRegisterError(AiowayError, KeyError): ...
