# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect
from abc import ABC
from typing import ClassVar

from aioway import registries
from aioway.errors import AiowayError

__all__ = ["Nargs"]


class Nargs(ABC):
    """
    The base class for ``Batch``, ``Poller``, and ``Op``.
    """

    ARGC: ClassVar[int]
    """
    Argument count of the current node.
    """

    @classmethod
    def _validate_nary_name(cls, key: str) -> None:
        """
        For `ARGC = k`, ``key`` must have `_k` suffix.
        """

        if key.endswith(f"_{cls.ARGC}"):
            return

        raise NaryNamingError(
            f"`ARGC={cls.ARGC}`, but {key=} does not have `_{cls.ARGC}` suffix."
        )

    @classmethod
    def _init_subclass(cls, base_cls: type["Nargs"], /, *, key: str = ""):
        """
        The shared ``__init_subclass__`` function for `Nargs` subclasses,
        including adding to registry and validating key name.
        """

        # Allow abstract classes, which would not be initialized,
        # to not define keys, as factories are used to store leaf nodes.
        if inspect.isabstract(cls):
            return

        # Impossible if `nargs_init_subclass` is only called in ``__init_subclass``.
        if not issubclass(cls, base_cls):
            raise NarySubclassError(
                "`nargs_init_subclass` must be called in `__init_subclass__`."
            )

        # Add to registry.
        registries.init_subclass(lambda: base_cls)(cls, key=key)

        # Ensure key name.
        cls._validate_nary_name(key)


class NaryNamingError(AiowayError, NameError): ...


class NarySubclassError(AiowayError, RuntimeError): ...
