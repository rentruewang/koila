# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import logging
from abc import ABC
from collections.abc import Iterable, Iterator
from typing import ClassVar

from aioway import registries
from aioway.blocks import Block
from aioway.errors import AiowayError

__all__ = ["Exec"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class Exec(Iterable[Block], ABC):
    """
    ``Exec`` is the graph / symbolic representation of an execution.

    An execution itself does not store state,
    but rather launches an iterator / cursor to iterate over the data.
    This design allows users to write genertors (``__iter__`` funciton),
    rather than iterators (``__next__`` function) with state management.
    """

    ARGC: ClassVar[int]
    """
    Argument count of the current node.
    """

    def __init_subclass__(cls, key: str = ""):
        # Allow abstract classes, which would not be initialized,
        # to not define keys, as factories are used to store leaf nodes.
        if inspect.isabstract(cls):
            return

        # Impossible if `nargs_init_subclass` is only called in ``__init_subclass``.
        if not issubclass(cls, Exec):
            raise ExecSubclassError(
                "`nargs_init_subclass` must be called in `__init_subclass__`."
            )

        # Add to registry.
        registries.init_subclass(lambda: Exec)(cls, key=key)

        # Ensure key name.
        cls._validate_nary_name(key)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Block]:
        """
        The ``__iter__`` method launches a new ``Iterator`` to loop over the inputs.
        Every call is creates / rebuilds brand new computation.

        Returns:
            A stream of ``Block``s.
        """

        ...

    @classmethod
    def _validate_nary_name(cls, key: str) -> None:
        """
        For `ARGC = k`, ``key`` must have `_k` suffix.
        """

        if key.endswith(f"_{cls.ARGC}"):
            return

        raise ExecNamingError(
            f"`ARGC={cls.ARGC}`, but {key=} does not have `_{cls.ARGC}` suffix."
        )


class ExecInitError(AiowayError, TypeError): ...


class ExecNamingError(AiowayError, NameError): ...


class ExecSubclassError(AiowayError, RuntimeError): ...
