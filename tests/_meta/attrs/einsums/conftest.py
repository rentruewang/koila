# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway._meta.attrs import EinsumParser


@pytest.fixture(scope="module")
def parser():
    return EinsumParser.init()
