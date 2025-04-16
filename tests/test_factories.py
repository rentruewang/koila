# Copyright (c) RenChu Wang - All Rights Reserved

import typing

from aioway import factories


@typing.no_type_check
def test_factory_init_subclass():
    class Base:
        __init_subclass__ = factories.init_subclass(lambda: Base)

    class A(Base, key="a"): ...

    class B(Base, key="b"): ...

    class C(Base, key="c"): ...

    fac = factories.of(Base)
    assert len(fac) == 3
    assert fac.keys() == set("abc")
    assert fac["a"] is A
    assert fac["b"] is B
    assert fac["c"] is C
