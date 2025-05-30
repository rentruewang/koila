# Copyright (c) AIoWay Authors - All Rights Reserved

from collections.abc import Callable, Iterator

import sympy as sym

_FUNCTIONS = [
    "x * 2",
    "x + 3",
    "x - 1",
    "x ** 2",
    "log(x)",
]


def funcs() -> Iterator[Callable[[float], float]]:
    for func in _FUNCTIONS:
        yield sym.lambdify("x", func)
