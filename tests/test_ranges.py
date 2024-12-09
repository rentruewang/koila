# Copyright (c) RenChu Wang - All Rights Reserved

import pytest

from aioway.previews import BoolChoice, FloatInterval, IntInterval, StrChoice


@pytest.mark.parametrize("range_class", [IntInterval, FloatInterval])
def test_range_fail_lower_gt_upper(range_class):
    with pytest.raises(ValueError):
        range_class(lower=1, upper=0)


@pytest.mark.parametrize("range_class", [IntInterval, FloatInterval])
def test_range_contains(range_class):
    range_checker = range_class(lower=1, upper=999)

    assert 1000 not in range_checker
    assert 555 in range_checker


@pytest.mark.parametrize("range_class", [IntInterval, FloatInterval])
def test_range_fail_default_not_in_range(range_class):
    with pytest.raises(ValueError):
        range_class(lower=0, upper=1, default=666)


@pytest.mark.parametrize("range_class", [IntInterval, FloatInterval])
def test_range_default(range_class):
    range_checker = range_class(lower=1, upper=999, default=666)
    assert range_checker.default == 666


@pytest.fixture
def bool_choice():
    return BoolChoice()


def test_bool_choice(bool_choice):
    assert True in bool_choice
    assert False in bool_choice

    assert 0 in bool_choice
    assert 1 in bool_choice

    assert -1 not in bool_choice
    assert 2 not in bool_choice


@pytest.fixture
def str_choice():
    return StrChoice(options=list("abcde"), default="a")


def test_string_choice(str_choice):
    assert "a" == str_choice.default
    assert "a" in str_choice
    assert "b" in str_choice
    assert "c" in str_choice
    assert "d" in str_choice
    assert "e" in str_choice
    assert "f" not in str_choice
