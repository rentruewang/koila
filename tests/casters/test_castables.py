# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

import pytest

from aioway.casters import Castable, Caster, TypeCastError

a_to_a = None


def b_to_b(b):
    assert isinstance(b, CastableB)

    return CastableB(source="identity")


def b_to_a(b):
    assert isinstance(b, CastableB)
    return CastableA(source="b")


def a_to_b(a):
    assert isinstance(a, CastableA)
    return CastableB(source="a")


class CastableBase(Castable):
    @classmethod
    def _caster(cls):
        return Caster(
            base=CastableBase,
            aliases=["a", "b"],
            klasses=[CastableA, CastableB],
            matrix=[
                [a_to_a, a_to_b],
                [b_to_a, b_to_b],
            ],
        )


@dcls.dataclass(frozen=True)
class CastableA(CastableBase):
    source: str = ""


@dcls.dataclass(frozen=True)
class CastableB(CastableBase):
    source: str = ""


@pytest.fixture
def cast_a():
    result = CastableA()
    assert isinstance(result, CastableBase)
    return result


@pytest.fixture
def cast_b():
    result = CastableB()
    assert isinstance(result, CastableBase)
    return result


def test_castable_has_astype(cast_a):
    assert callable(cast_a.astype)


def test_undefined_cast_to_self(cast_a):
    result = cast_a.astype("a")
    assert result is cast_a


def test_a_to_b(cast_a):
    result = cast_a.astype("b")
    assert result.source == "a"
    assert isinstance(result, CastableB)


def test_b_to_a(cast_b):
    result = cast_b.astype("a")
    assert result.source == "b"
    assert isinstance(result, CastableA)


def test_b_to_b(cast_b):
    result = cast_b.astype("b")
    assert isinstance(result, CastableB)
    assert result.source == "identity"
