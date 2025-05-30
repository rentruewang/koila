# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway import factories
from aioway.procs import (
    CallbackProc,
    CtxProc,
    LoggingProc,
    MemoProc,
    OpaqueProc,
    Proc,
    StackProc,
)


@pytest.fixture(scope="module")
def proc_factory():
    return factories.of(Proc)


def _proc_key_cls_param():
    yield "CALLBACK", CallbackProc
    yield "CTX", CtxProc
    yield "LOGGING", LoggingProc
    yield "MEMO", MemoProc
    yield "OPAQUE", OpaqueProc
    yield "STACK", StackProc


@pytest.fixture(scope="module", params=_proc_key_cls_param())
def proc_key_cls(request) -> tuple[str, type]:
    return request.param


def test_proc_factory_getitem(proc_factory, proc_key_cls):
    key, cls = proc_key_cls
    assert proc_factory[key] is cls
