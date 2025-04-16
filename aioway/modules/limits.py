# Copyright (c) RenChu Wang - All Rights Reserved

__all__ = ["LimitSet"]

import dataclasses as dcls
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import Self

from sympy import Set

from aioway.errors import AiowayError


@dcls.dataclass(frozen=True)
class LimitSet(Mapping[str, Set]):
    """
    ``LimitSet`` is essentially a mapping of key to constraints,
    informing the compiler what arguments are avaialble,
    and what their valid values are.
    """

    params: dict[str, Set]
    """
    The mapping from a valid value to their constraints.
    """

    def __post_init__(self) -> None:
        for constraint in self.params.values():
            if not isinstance(constraint, Set):
                raise LimitSetTypeError(
                    f"Constraint given: {constraint} is not a `sympy.Set`"
                )

    def __contains__(self, key: object) -> bool:
        return key in self.params

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, key: str) -> Set:
        return self.params[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.params

    def keys(self) -> KeysView[str]:
        return self.params.keys()

    def values(self) -> ValuesView[Set]:
        return self.params.values()

    def items(self) -> ItemsView[str, Set]:
        return self.params.items()

    @classmethod
    def from_params(cls, **constraints: Set) -> Self:
        return cls(params=constraints)


class LimitSetTypeError(AiowayError, TypeError): ...
