# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway import _registries


@typing.no_type_check
def test_factory_init_subclass():
    class Base:
        __init_subclass__ = _registries.init_subclass(lambda: Base)

    class A(Base, key="a"): ...

    class B(Base, key="b"): ...

    class C(Base, key="c"): ...

    fac = _registries.of(Base)
    assert len(fac) == 3
    assert fac.keys() == set("abc")
    assert fac["a"] is A
    assert fac["b"] is B
    assert fac["c"] is C
