# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.casters import Castable, Caster


class UncastableBase(Castable):
    @classmethod
    def _caster(cls):
        return Caster(
            base=UncastableBase,
            aliases=["a", "b"],
            klasses=[UnCastableA, UnCastableB],
            matrix=[[None, None], [None, None]],
        )


class UnCastableA(UncastableBase): ...


class UnCastableB(UncastableBase): ...


@pytest.fixture
def no_cast_a():
    result = UnCastableA()
    assert isinstance(result, UncastableBase)
    return result


def test_castable_has_cast(no_cast_a):
    assert callable(no_cast_a.cast)


def test_uncastable_casts_to_self(no_cast_a):
    no_cast_a.cast("a")


def test_uncastable_to_other(no_cast_a):
    with pytest.raises(TypeError):
        no_cast_a.cast("b")
