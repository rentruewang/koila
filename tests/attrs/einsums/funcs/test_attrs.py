# Copyright (c) AIoWay Authors - All Rights Reserved

import json
from pathlib import Path

import pytest

from aioway.attrs import EinsumAttr, EinsumDevice, EinsumDType, EinsumName, EinsumShape


def _test_einsum_func_init(name: str, einsum_func: type[EinsumAttr]):
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
            with pytest.raises(ValueError):
                # Even the failing cases are supplied with parsable inputs.
                einsum = parser(failing)
                einsum_func(einsum)

    return TestClass


TestEinsumName = _test_einsum_func_init("names", EinsumName)
TestEinsumShape = _test_einsum_func_init("shapes", EinsumShape)
TestEinsumDevice = _test_einsum_func_init("devices", EinsumDevice)
TestEinsumDType = _test_einsum_func_init("dtypes", EinsumDType)
