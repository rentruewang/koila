# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from typing import Self

from aioway._errors import AiowayError

__all__ = ["SchemaType"]


@dcls.dataclass(frozen=True, repr=False)
class SchemaType:
    """
    ``SchemaType`` is the types of a normal SQL database schema.

    Note:
        Extend this class to handle real schema types.
    """

    name: str
    """
    The type of the current schema.
    """

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def parse(cls, name: str) -> Self:
        return cls(name)


class SchemaTypeError(AiowayError, ValueError): ...
