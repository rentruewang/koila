# Copyright (c) AIoWay Authors - All Rights Reserved

from collections.abc import Generator

import pytest

from aioway import variants
from aioway.attrs import Attr, AttrSet, Device, DeviceSet, DTypeSet, Shape, ShapeSet
from aioway.variants import ParamList, Signature, SignatureRegistry


def _signatures() -> Generator[list[Signature]]:
    yield [
        Signature.ufunc1(Device),
        Signature.ufunc2(Device),
    ]

    yield [
        Signature.ufunc1(Device),
        Signature.ufunc2(Device),
        Signature.ufunc1(Shape),
        Signature.ufunc2(Shape),
    ]
    yield [
        Signature.ufunc1(Device),
        Signature.ufunc1(DeviceSet),
    ]
    yield [
        Signature.ufunc1(Attr),
        Signature.ufunc1(AttrSet),
    ]
    yield [
        Signature.ufunc1(ShapeSet),
        Signature.ufunc1(DeviceSet),
        Signature.ufunc1(DTypeSet),
        Signature.ufunc1(AttrSet),
    ]


@pytest.fixture(params=_signatures(), scope="module")
def signature_list(request) -> list[ParamList]:
    return request.param


@pytest.fixture(scope="module")
def registry(signature_list) -> SignatureRegistry:
    "The (partial) registry used for testing."
    return variants.default_registry().select(*signature_list)


def test_registry_select(registry, signature_list):
    assert registry.signatures == set(signature_list)
