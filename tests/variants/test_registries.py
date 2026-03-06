# Copyright (c) AIoWay Authors - All Rights Reserved

from collections.abc import Generator

import pytest

from aioway import variants
from aioway.attrs import Attr, AttrSet, Device, DeviceSet, DTypeSet, Shape, ShapeSet
from aioway.variants import ParamList, SignatureRegistry


def _param_lists() -> Generator[list[ParamList]]:
    yield [
        ParamList(Device),
        ParamList(Device, Device),
    ]

    yield [
        ParamList(Device),
        ParamList(Device, Device),
        ParamList(Shape),
        ParamList(Shape, Shape),
    ]
    yield [
        ParamList(Device),
        ParamList(DeviceSet),
    ]
    yield [
        ParamList(Attr),
        ParamList(AttrSet),
    ]
    yield [
        ParamList(ShapeSet),
        ParamList(DeviceSet),
        ParamList(DTypeSet),
        ParamList(AttrSet),
    ]


@pytest.fixture(params=_param_lists(), scope="module")
def param_lists(request) -> list[ParamList]:
    return request.param


@pytest.fixture(scope="module")
def registry(param_lists) -> SignatureRegistry:
    "The (partial) registry used for testing."
    return variants.default_registry().select(*param_lists)


def test_registry_select(registry, param_lists):
    assert registry.signatures == set(param_lists)
