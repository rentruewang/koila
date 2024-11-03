# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from aioway.common import LazyStr


def test_strings():
    s = "hello world"
    ls = LazyStr(s)

    assert str(ls) == s
    assert repr(ls) == s
    assert ls() == s

    ls = LazyStr(lambda: s)

    assert str(ls) == s
    assert repr(ls) == s
    assert ls() == s
