# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import KeysView, Sequence, Set

__all__ = ["SeqKeysView", "SetKeysView"]


@dcls.dataclass(frozen=True)
class _ContainerKeysView[C: Sequence[str] | Set[str]](KeysView[str]):
    seq: C

    @typing.override
    def __contains__(self, key: object) -> bool:
        return key in self.seq

    @typing.override
    def __iter__(self):
        yield from self.seq


class SeqKeysView(_ContainerKeysView[Sequence[str]]): ...


class SetKeysView(_ContainerKeysView[set[str]]): ...
