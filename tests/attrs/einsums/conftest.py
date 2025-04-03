# Copyright (c) RenChu Wang - All Rights Reserved

import pytest

from aioway.attrs import EinsumParser


@pytest.fixture(scope="module")
def parser():
    return EinsumParser.init()
