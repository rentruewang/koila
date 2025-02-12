# Copyright (c) RenChu Wang - All Rights Reserved

import json
from pathlib import Path

import pytest

from aioway.schemas import Device


def example_devices() -> list[str]:
    with open(Path(__file__).parent / "devices.json") as f:
        return json.load(f)


@pytest.mark.parametrize("device", example_devices())
def test_devices_eq(device):
    assert Device(device) == device
