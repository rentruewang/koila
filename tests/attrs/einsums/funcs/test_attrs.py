# Copyright (c) RenChu Wang - All Rights Reserved

import json
from pathlib import Path

import pytest

from aioway.attrs import (
    EinsumAttrFunc,
    EinsumDevice,
    EinsumDType,
    EinsumMap,
    EinsumShape,
)


def _test_einsum_func_init(name: str, einsum_func: type[EinsumAttrFunc]):
    with open(Path(__file__).parent / f"{name}.json") as f:
        data = json.load(f)
        passing_data = data["pass"]
        failing_data = data["fail"]

    class TestClass:
        @pytest.fixture(scope="class", params=passing_data)
        def passing(self, request) -> str:
            return request.param

        @pytest.fixture(scope="class", params=failing_data)
        def failing(self, request) -> str:
            return request.param

        def test_passing(self, passing, parser):
            einsum = parser(passing)

            # Only verify that the checks all pass, which are in `__post_init__`.
            func = einsum_func(einsum)
            assert func == passing

        def test_failing(self, failing, parser):
            # Even the failing cases are supplied with parsable inputs.
            einsum = parser(failing)

            with pytest.raises(ValueError):
                einsum_func(einsum)

    return TestClass


TestEinsumMap = _test_einsum_func_init("maps", EinsumMap)
TestEinsumShape = _test_einsum_func_init("shapes", EinsumShape)
TestEinsumDevice = _test_einsum_func_init("devices", EinsumDevice)
TestEinsumDType = _test_einsum_func_init("dtypes", EinsumDType)
