# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

from sympy import Set

from aioway._errors import AiowayError

from .specs import Spec

__all__ = ["SympySetSpec"]


@dcls.dataclass(frozen=True)
class SympySetSpec(Spec):
    """
    ``SympySetSpec`` is a spec that checks if an object satisfies a sympy set.
    """

    sym_set: Set
    """
    The sympy set to check against.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.sym_set, Set):
            raise SympyTypeError(f"SympySetSpec given: {self.sym_set} is not a `Set`.")

    def __contains__(self, obj: object) -> bool:
        return obj in self.sym_set

    def __str__(self) -> str:
        return str(self.sym_set)


class SympyTypeError(AiowayError, TypeError): ...
