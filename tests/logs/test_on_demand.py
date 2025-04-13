# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import logging

import pytest

from aioway import logs


def _logger_methods():
    logger = logging.getLogger(__name__)

    yield logger.debug
    yield logger.info
    yield logger.warning
    yield logger.error
    yield logger.critical


@pytest.fixture(scope="module", params=_logger_methods())
def passing_case(request):
    return request.param


def _non_callable():
    yield 1
    yield 2.5
    yield "hello"
    yield logging.info


def _non_bound_methods():
    for item in _non_callable():
        yield lambda: item


def _non_logger_bound_methods():
    for item in _non_callable():

        @dcls.dataclass(frozen=True)
        class AdHoc:
            def f(self):
                return item

        yield AdHoc().f


def _failing_cases():
    yield from _non_callable()
    yield from _non_bound_methods()
    yield from _non_logger_bound_methods()


@pytest.fixture(scope="module", params=_failing_cases())
def failing_case(request):
    return request.param


def test_on_demand_wrap_logger_methods(passing_case):
    logs.on_demand(passing_case)


def test_on_demand_fail_everything_else(failing_case):
    with pytest.raises(TypeError):
        logs.on_demand(failing_case)
