# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Hashable, Iterator, MutableMapping


@typing.final
@dcls.dataclass(frozen=True)
class Factory[T: Hashable](MutableMapping[T, dict[str, T]]):
    registry: dict[T, dict[str, T]] = dcls.field(default_factory=dict)

    def __iter__(self) -> Iterator[T]:
        return iter(self.registry)

    def __len__(self) -> int:
        return len(self.registry)

    def __getitem__(self, key: T) -> dict[str, T]:
        return self.registry[key]

    def __setitem__(self, key: T, val: dict[str, T], /) -> None:
        self.registry[key] = val

    def __delitem__(self, key: T, /) -> None:
        del self.registry[key]

    @classmethod
    def of(cls, typ: T) -> dict[str, T]:
        factory = cls()

        # If this is already registered.
        if typ in factory:
            return factory[typ]

        # Register the registry.
        registry: dict[str, T] = {}
        factory[typ] = registry
        return registry
