# Copyright (c) RenChu Wang - All Rights Reserved

import pytest
from torch.nn import Linear


@pytest.fixture
def linear():
    return Linear(10, 5)
