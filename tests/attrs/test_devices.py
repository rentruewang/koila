# Copyright (c) AIoWay Authors - All Rights Reserved

import json
from pathlib import Path

import pytest
import torch

from aioway.attrs import Device


def example_devices() -> list[str]:
    with open(Path(__file__).parent / "devices.json") as f:
        return json.load(f)


@pytest.fixture(params=example_devices(), scope="module")
def device(request):
    return request.param


def test_devices_eq(device):
    assert Device(device) == device
    assert Device(device) == torch.device(device)
    assert Device.parse(device) == torch.device(device)
