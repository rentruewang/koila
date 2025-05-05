# Copyright (c) RenChu Wang - All Rights Reserved

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.nn import Linear

from aioway.attrs import Attr, AttrSet
from aioway.execs import ExecTracer
from aioway.frames import TensorDictFrame


@pytest.fixture
def linear():
    return Linear(10, 5)


@pytest.fixture
def module(linear):
    return TensorDictModule(linear, in_keys="a", out_keys="b")


@pytest.fixture
def module_attrs():
    return AttrSet(
        {
            "a": Attr.parse(
                {
                    "dtype": "float32",
                    "shape": [11, 10],
                    "device": "cpu",
                }
            ),
            "b": Attr.parse(
                {
                    "dtype": "float32",
                    "shape": [11, 5],
                    "device": "cpu",
                }
            ),
        }
    )


@pytest.fixture
def td():
    return TensorDict({"a": torch.randn(11, 10)}, batch_size=[11], device="cpu")


@pytest.fixture
def frame(td):
    return TensorDictFrame.from_dict(td)


@pytest.fixture
def tracer(frame):
    return ExecTracer.create("FRAME", frame, {"batch_size": 4})
