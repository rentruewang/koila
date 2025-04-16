# Copyright (c) RenChu Wang - All Rights Reserved

import pytest

from aioway import factories
from aioway.execs import (
    Exec,
    FilterExprExec,
    FilterPredExec,
    FrameStreamExec,
    MapExec,
    MatrixJoinExec,
    ProjectExec,
    RawIteratorExec,
    RenameExec,
    ZipExec,
)


@pytest.fixture(scope="module")
def exec_factory():

    return factories.of(Exec)


def _exec_key_cls_param():
    yield "MATRIX_JOIN", MatrixJoinExec
    yield "ZIP", ZipExec
    yield "FRAME_STREAM", FrameStreamExec
    yield "RAW_ITER", RawIteratorExec
    yield "FILTER_PRED", FilterPredExec
    yield "FILTER_EXPR", FilterExprExec
    yield "MAP", MapExec
    yield "PROJECT", ProjectExec
    yield "RENAME", RenameExec


@pytest.fixture(scope="module", params=_exec_key_cls_param())
def exec_key_cls(request) -> tuple[str, type]:
    return request.param


def test_exec_factory_getitem(exec_factory, exec_key_cls):
    key, cls = exec_key_cls
    assert exec_factory[key] is cls
