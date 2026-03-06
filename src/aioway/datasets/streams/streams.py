# Copyright (c) AIoWay Authors - All Rights Reserved

"The ``Stream`` interfaces live here."

import abc
import dataclasses as dcls
import functools
import inspect
import typing
from abc import ABC
from collections.abc import Generator, Iterator
from typing import ClassVar, Self

from aioway import variants
from aioway.attrs import AttrSet
from aioway.batches import Chunk
from aioway.variants import ParamList

from ..datasets import Dataset, DatasetViewTypes

__all__ = ["Stream", "StreamState", "Stream0", "Stream1", "Stream2"]


@dcls.dataclass
class StreamState:
    """
    The mutable stream state.

    This is created because ``Stream`` subclasses from a frozen ``dataclass``,
    so the stream state is created to manage mutable parts of the ``Stream``.

    Subclasses of ``Stream`` should also subclass from ``StreamState``.
    """

    idx: int = 0
    "How many steps have been called."

    def step(self):
        self.idx += 1

    @property
    def started(self) -> bool:
        """
        Shortcut function to check if ``self.idx == 0``.
        """

        return self.idx != 0


@dcls.dataclass(frozen=True)
class Stream(Iterator[Chunk], Dataset, ABC):
    """
    ``Stream`` produces a stream of batches of data, in the form of ``TensorDict``s,
    everytime ``__next__`` is called on it, a ``TensorDict`` is yielded.

    ``Stream`` is a stateful operation, compared to the previous implementations,
    it is an external iterator, supporting state inspection, simplifying debugging.
    """

    _SIGNATURE: ClassVar[ParamList]
    "The signature of the current class."

    __match_args__: ClassVar[tuple[str, ...]]
    """
    A ``Stream`` should be able to be decomposed with ``match`` statements.
    """

    @classmethod
    def __init_subclass__(cls, key: str = "") -> None:
        cls.__register_subclass(key)

    @typing.override
    def __iter__(self) -> Self:
        """
        ``__iter__`` allows ``Stream``s to be used in ``for`` loops.

        As it returns ``self``, re-use carefully.
        """

        return self

    @typing.final
    @typing.override
    def __next__(self) -> Chunk:
        """
        ``__next__`` allows ``Stream``s to be used in ``for`` loops.
        """

        if (result := self._read()).attrs != self.attrs:
            raise TypeError(f"Schema mismatch: {result.attrs=}, {self.attrs=}.")

        self.state.step()
        return result

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """
        The length of the current ``Stream``.
        Does not change when the ``Stream`` is being iterated over.
        """

        ...

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet:
        """
        The schema for the current ``Stream``.
        """

        ...

    @abc.abstractmethod
    def _read(self) -> Chunk:
        """
        Compute the next batch (a ``Chunk``).

        An exception raised here would be translated to ``StopIteration``.

        Note:

            The name ``read`` is inspired by the original release of clickhouse.
            I think this is a great name for a as a method on ``Stream`` to poll data,
            giving it a natural feeling like how we normally work with files.

        """

        ...

    @typing.final
    def children(self) -> list["Stream"]:
        """
        Yields the children (dependencies of the current ``Stream``).

        Does not yield ``self``.

        Yields:
            Some ``Stream``s that would be evaluated if the current is running.
        """

        children_list = list(self._children())

        if len(children_list) != self.argc():
            raise AssertionError(
                f"Number of children must match signature {self._SIGNATURE}."
            )

        for child in children_list:
            assert child is not self, "A `Stream` cannot depend on itself!"

        return children_list

    @abc.abstractmethod
    def _children(self) -> Generator["Stream"]:
        """
        ``Stream``'s children, the dependent ``Stream``s that would also be evaluated
        when calling ``__next__`` on the current ``Stream``.
        """

        ...

    @functools.cached_property
    def state(self) -> StreamState:
        """
        The state of the stream. Should be a field, but a ``cached_property``,
        because if it has a default value it would make subclassing difficult.
        """
        return StreamState()

    @property
    def idx(self) -> int:
        """
        The number of batches completed..
        """

        return self.state.idx

    @property
    def started(self) -> bool:
        """
        Shortcut function to check if ``self.idx == 0``.
        """

        return self.state.started

    @classmethod
    @typing.override
    def view_types(cls):
        from .views import StreamColumnView, StreamSelectView

        return DatasetViewTypes(column=StreamColumnView, select=StreamSelectView)

    @classmethod
    def __register_subclass(cls, key: str):
        # Don't do anything for abstract classes.
        if inspect.isabstract(cls):
            return

        # Concrete subclass must have key.
        if not key:
            raise KeyError(f"Concrete class {cls} should provide a key.")

        # Register.
        variants.register(cls._SIGNATURE, key)(cls)

    @classmethod
    def argc(cls) -> int:
        return len(cls._SIGNATURE)


class Stream0(Stream, ABC):
    _SIGNATURE = ParamList()


class Stream1(Stream, ABC):
    _SIGNATURE = ParamList(Stream)


class Stream2(Stream, ABC):
    _SIGNATURE = ParamList(Stream, Stream)
