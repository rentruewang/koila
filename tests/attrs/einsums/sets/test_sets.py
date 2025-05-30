# Copyright (c) AIoWay Authors - All Rights Reserved

import json
from pathlib import Path
from typing import TypedDict

import pytest

from aioway.attrs import EinsumAttrSet


class AttrSetDict(TypedDict):
    names: str
    dtypes: str
    shapes: str
    devices: str


def load_data(split: str):
    with open(Path(__file__).parent / f"{split}.json") as f:
        return json.load(f)


@pytest.fixture(params=load_data("passing"))
def passing(request) -> AttrSetDict:
    return request.param


def test_passing(passing):
    EinsumAttrSet.parse(**passing)
