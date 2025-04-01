# Copyright (c) RenChu Wang - All Rights Reserved


from aioway.factories import Factory


def test_factory_simple():
    fac: Factory = Factory()
    assert isinstance(fac, Factory)

    class A:
        pass

    fac["a"] = A

    assert fac["a"] == A
    assert fac[A] == "a"

    assert len(fac) == 1

    class B:
        pass

    fac["b"] = B

    assert fac["b"] == B
    assert fac[B] == "b"

    assert len(fac) == 2
