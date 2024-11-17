# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections import defaultdict as DefaultDict
from typing import Generic, Self, TypeVar

from aioway.relalg import Relation

from .previews import Preview

_T = TypeVar("_T")


@dcls.dataclass(frozen=True)
class Registry(Generic[_T]):
    previews: list[Preview[_T]] = dcls.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.previews)

    def __getitem__(self, key: int) -> Preview[_T]:
        return self.previews[key]

    def register(self, preview: Preview[_T]) -> Preview[_T]:
        self.previews.append(preview)
        return preview

    def filter_by_relation(self, relation: type[Relation]) -> Self:
        return type(self)([p for p in self.previews if p.relation is relation])

    def split_by_relation(self) -> dict[type[Relation], Self]:
        dd: dict[type[Relation], Self] = DefaultDict(type(self))

        for p in self.previews:
            dd[p.relation].previews.append(p)

        return dd
