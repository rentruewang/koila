# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway import registries
from aioway.execs import (
    EchoExec,
    Execution,
    FilterExprExec,
    FilterPredExec,
    FrameExec,
    IteratorExec,
    MapExec,
    ModuleExec,
    NestedLoopExec,
    ProjectExec,
    RenameExec,
    ZipExec,
)


@pytest.fixture(scope="module")
def exec_factory():
    return registries.of(Execution)


def _exec_key_cls_param():
    yield "NESTED_LOOP", NestedLoopExec
    yield "ZIP", ZipExec
    yield "FRAME", FrameExec
    yield "ITER", IteratorExec
    yield "FILTER_PRED", FilterPredExec
    yield "FILTER_EXPR", FilterExprExec
    yield "MAP", MapExec
    yield "MODULE", ModuleExec
    yield "PROJECT", ProjectExec
    yield "RENAME", RenameExec
    yield "ECHO", EchoExec


@pytest.fixture(scope="module", params=_exec_key_cls_param())
def exec_key_cls(request) -> tuple[str, type]:
    return request.param


def test_exec_factory_getitem(exec_factory, exec_key_cls):
    key, cls = exec_key_cls
    assert exec_factory[key] is cls
