# Copyright (c) AIoWay Authors - All Rights Reserved

import functools
import json
from pathlib import Path

import pytest

from aioway.attrs import EinsumSignature


@functools.cache
def _load_cases():
    file = Path(__file__).parent / "einsums.json"

    with file.open() as f:
        return json.load(f)


@pytest.fixture(scope="module", params=_load_cases()["pass"])
def passing(request) -> str:
    return request.param


@pytest.fixture(scope="module", params=_load_cases()["fail"])
def failing(request) -> str:
    return request.param


def test_einsum_parser_passing(passing, parser):
    assert isinstance(passing, str)

    parsed = parser(passing)
    assert isinstance(parsed, EinsumSignature)

    # `Einsum` would deal with the comparison.
    assert parsed == passing


def test_failing(failing, parser):
    assert isinstance(failing, str)

    with pytest.raises(ValueError):
        parser(failing)
