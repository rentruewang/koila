# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from collections.abc import Iterator, Mapping
from typing import NamedTuple, Self

from aioway._errors import AiowayError

from .specs import Spec

__all__ = ["SpecSet"]


class NamedParam(NamedTuple):
    "A named tuple for packing the name and spec."

    name: str
    "Name of the parameter."

    spec: Spec
    "Spec for the parameter."


@dcls.dataclass(frozen=True)
class SpecSet:
    """
    ``SpecSet`` is essentially a mapping of key to constraints,
    informing the compiler what arguments are avaialble,
    and what their valid values are.
    """

    specs: dict[str, Spec]
    """
    The mapping from a valid value to their constraints.
    """

    def __post_init__(self) -> None:
        for val in self.specs.values():
            if not isinstance(val, Spec):
                raise SpecSetTypeError(f"Constraint given: {val} is not a `Spec`.")

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, key: str) -> Spec:
        return self.specs[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.specs

    def __contains__(self, mapping: object) -> bool:
        # `SpecSet` specializes in checking if a mapping is valid.
        if not isinstance(mapping, Mapping):
            return False

        # Mapping must have all keys provided.
        if mapping.keys() != self.specs.keys():
            return False

        for key, val in mapping.items():
            # Check if the value is in the spec.
            if val not in self.specs[key]:
                return False

        return True

    def params(self) -> Iterator[NamedParam]:
        for name, spec in self.specs.items():
            yield NamedParam(name=name, spec=spec)

    @classmethod
    def from_specs(cls, **constraints: Spec) -> Self:
        return cls(specs=constraints)


class SpecSetTypeError(AiowayError, TypeError): ...
