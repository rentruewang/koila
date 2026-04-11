# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

__all__ = ["UFunc1", "UFunc2", "AnyUFunc1", "AnyUFunc2"]


class Named(typing.Protocol):
    __qualname__: str


class UFunc1(Named, typing.Protocol):
    def __call__[T](self, item: T, /) -> T: ...


class UFunc2(Named, typing.Protocol):
    def __call__[T](self, left: T, right: T, /) -> T: ...


class AnyUFunc1[T](Named, typing.Protocol):
    def __call__(self, item, /) -> T: ...


class AnyUFunc2[T](Named, typing.Protocol):
    def __call__(self, left: typing.Any, right: typing.Any, /) -> T: ...
