# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import difflib
import logging
import typing
from collections.abc import MutableMapping
from typing import Any

from aioway._errors import AiowayError

__all__ = ["Registry", "RegistryKeyError"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True, repr=False)
class Registry[T](MutableMapping[str, T]):
    """
    ``Registry`` supports looking up a collection of items by their names.

    You will not be able to overwrite classes (2 classes with the same key),
    an error would be raised in that case.
    """

    registry: dict[str, T] = dcls.field(default_factory=dict)
    """
    The registry mapping from a key to a value.
    """

    @typing.override
    def __len__(self) -> int:
        return len(self.registry)

    @typing.override
    def __contains__(self, obj: Any) -> bool:
        return obj in self.registry

    @typing.override
    def __iter__(self):
        yield from self.registry

    @typing.override
    def __getitem__(self, key: str, /) -> T:
        LOGGER.debug("Getting item for key: %s", key)
        self._raise_error_not_found(key)
        return self.registry[key]

    @typing.override
    def __setitem__(self, key: str, item: T, /) -> None:
        if key in self:
            raise RegistryKeyError(
                f"Trying to insert key: {key} and item: {item} "
                f"but key is already used by item: {self[key]}"
            )

        self.registry[key] = item

    @typing.override
    def __delitem__(self, key: str) -> None:
        self._raise_error_not_found(key)
        del self.registry[key]

    def __repr__(self) -> str:
        return repr(self.registry)

    def _raise_error_not_found(self, key: str, /) -> None:
        """
        Raise an error and suggest an alternative,
        with the closest name if not found.
        """

        if key in self:
            return

        closest = difflib.get_close_matches(key, self)

        msg = f"{key=} not found in registry."

        if len(closest):
            candidates = ", ".join(f"'{close}'" for close in closest)
            msg = f"{msg} Do you mean one of: [{candidates}]"

        raise RegistryKeyError(msg)


class RegistryKeyError(AiowayError, KeyError): ...
