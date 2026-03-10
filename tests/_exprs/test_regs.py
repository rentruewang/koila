# Copyright (c) AIoWay Authors - All Rights Reserved

from collections.abc import Generator

import pytest

from aioway import _exprs
from aioway._exprs import OpSign, SignatureRegistry, TypeList
from aioway.attrs import Attr, AttrSet, Device, DeviceSet, DTypeSet, Shape, ShapeSet


def _signatures() -> Generator[list[OpSign]]:
    yield [
        OpSign.ufunc1(Device),
        OpSign.ufunc2(Device),
    ]

    yield [
        OpSign.ufunc1(Device),
        OpSign.ufunc2(Device),
        OpSign.ufunc1(Shape),
        OpSign.ufunc2(Shape),
    ]
    yield [
        OpSign.ufunc1(Device),
        OpSign.ufunc1(DeviceSet),
    ]
    yield [
        OpSign.ufunc1(Attr),
        OpSign.ufunc1(AttrSet),
    ]
    yield [
        OpSign.ufunc1(ShapeSet),
        OpSign.ufunc1(DeviceSet),
        OpSign.ufunc1(DTypeSet),
        OpSign.ufunc1(AttrSet),
    ]


@pytest.fixture(params=_signatures(), scope="module")
def signature_list(request) -> list[TypeList]:
    return request.param


@pytest.fixture(scope="module")
def registry(signature_list) -> SignatureRegistry:
    "The (partial) registry used for testing."
    return _exprs.default_registry().select(*signature_list)


def test_registry_select(registry, signature_list):
    assert registry.signatures == set(signature_list)
