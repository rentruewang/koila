# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable

__all__ = ["format_function", "dcls_no_eq"]


def format_function(func: Callable, *args: typing.Any, **kwargs: typing.Any) -> str:
    "Format the function into readable string, mimicking signature in python."

    args_builder: list[str] = []

    # Add positional arguments.
    if args:
        args_builder.extend(f"{arg!r}" for arg in args)

    # Add keyword arguments.
    if kwargs:
        args_builder.extend(f"{k!s}={v!r}" for k, v in kwargs.items())

    args_str = ", ".join(args_builder)
    func_str = func.__qualname__
    return f"{func_str}({args_str})"


@typing.dataclass_transform(eq_default=False)
def dcls_no_eq[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(eq=False)(cls)
    return result
