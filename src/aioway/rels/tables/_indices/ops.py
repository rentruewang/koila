# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC

__all__ = [
    "IndexPlan",
    "IndexEq",
    "IndexNe",
    "IndexAnn",
    "IndexGt",
    "IndexGe",
    "IndexLt",
    "IndexLe",
]


class IndexPlan(ABC):
    """
    `IndexPlan` is the dynamic feature we would want to retrieve from an ``Index``.
    """

    @abc.abstractmethod
    def __str__(self) -> str: ...


@dcls.dataclass(frozen=True, repr=False)
class IndexEq(IndexPlan):
    @typing.override
    def __str__(self):
        return "=="


@dcls.dataclass(frozen=True, repr=False)
class IndexNe(IndexPlan):
    @typing.override
    def __str__(self):
        return "!="


@dcls.dataclass(frozen=True, repr=False)
class IndexAnn(IndexPlan):
    """
    The approximate nearest neighbors for indices.
    """

    k: int
    """
    The top k items to use.
    """

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(
                f"Number of nearest neighbors must be a natural number. Got {self.k}"
            )

    @typing.override
    def __str__(self):
        return f"~{self.k}"


@dcls.dataclass(frozen=True, repr=False)
class IndexGt(IndexPlan):
    @typing.override
    def __str__(self):
        return "<"


@dcls.dataclass(frozen=True, repr=False)
class IndexGe(IndexPlan):
    @typing.override
    def __str__(self):
        return "<="


@dcls.dataclass(frozen=True, repr=False)
class IndexLt(IndexPlan):
    @typing.override
    def __str__(self):
        return ">"


@dcls.dataclass(frozen=True, repr=False)
class IndexLe(IndexPlan):
    @typing.override
    def __str__(self):
        return ">="
