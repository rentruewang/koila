# Copyright (c) RenChu Wang - All Rights Reserved

from collections import defaultdict as DefaultDict

import pytest

from aioway.factories import Factory, Registry


@pytest.fixture(scope="function")
def registry() -> Registry:
    return DefaultDict(Factory)


def test_factory_simple(registry):
    class A:
        pass

    fac: Factory = Factory.of(A, reg=registry)
    assert isinstance(fac, Factory)

    assert A in registry.keys()
    assert len(registry) == 1


def test_factory_registry(registry):
    class A:
        def __init_subclass__(cls, key: str):
            fac: Factory = Factory.of(A, reg=registry)
            fac[key] = cls

    Factory.of(A, reg=registry)

    assert A in registry.keys()
    assert len(registry) == 1

    @Factory.register(reg=registry)
    class B:
        pass

    assert B in registry.keys()
    assert len(registry) == 2

    class C(A, key="c"):
        pass

    assert Factory.of(A, reg=registry)["c"] is C
    assert Factory.of(A, reg=registry)("c") is C
