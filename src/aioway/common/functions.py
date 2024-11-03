# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


def wraps(
    function: Callable[_P, _T], /
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    This is equivalent to `functools.wraps` with proper typing support.
    """

    # ``function`` is the original function with documentation,
    # and ``func`` is the function to wrap / decorate.

    if not callable(function):
        raise ValueError("Cannot wrap an object not callable.")

    def factory(func: Callable[_P, _T]) -> Callable[_P, _T]:
        # @wraps(a)
        # def b(...) -> ...: ...
        # would become factory(b)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            return func(*args, **kwargs)

        return wrapper

    return factory
