# Copyright (c) AIoWay Authors - All Rights Reserved

from typing import Any, Protocol

__all__ = ["UFunc1", "UFunc2", "AnyUFunc1", "AnyUFunc2"]


class Named(Protocol):
    __qualname__: str


class UFunc1(Named, Protocol):
    def __call__[T](self, item: T, /) -> T: ...


class UFunc2(Named, Protocol):
    def __call__[T](self, left: T, right: T, /) -> T: ...


class AnyUFunc1[T](Named, Protocol):
    def __call__(self, item, /) -> T: ...


class AnyUFunc2[T](Named, Protocol):
    def __call__(self, left: Any, right: Any, /) -> T: ...
