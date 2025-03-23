# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC

from aioway.errors import AiowayError

__all__ = [
    "IndexOp",
    "IndexEq",
    "IndexNe",
    "IndexReq",
    "IndexGt",
    "IndexGe",
    "IndexLt",
    "IndexLe",
]


class IndexOp(ABC):
    @abc.abstractmethod
    def __str__(self) -> str: ...


@dcls.dataclass(frozen=True, repr=False)
class IndexEq(IndexOp):
    def __str__(self):
        return "=="


@dcls.dataclass(frozen=True, repr=False)
class IndexNe(IndexOp):
    def __str__(self):
        return "!="


@dcls.dataclass(frozen=True, repr=False)
class IndexReq(IndexOp):
    k: int

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise IndexReqKError(
                f"Number of nearest neighbors must be a natural number. Got {self.k}"
            )

    def __str__(self):
        return f"~{self.k}"


@dcls.dataclass(frozen=True, repr=False)
class IndexGt(IndexOp):
    def __str__(self):
        return "<"


@dcls.dataclass(frozen=True, repr=False)
class IndexGe(IndexOp):
    def __str__(self):
        return "<="


@dcls.dataclass(frozen=True, repr=False)
class IndexLt(IndexOp):
    def __str__(self):
        return ">"


@dcls.dataclass(frozen=True, repr=False)
class IndexLe(IndexOp):
    def __str__(self):
        return ">="


class IndexReqKError(AiowayError, ValueError): ...
