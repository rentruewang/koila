# Copyright (c) RenChu Wang - All Rights Reserved

import random

import pytest
import torch
from numpy import random as npr


@pytest.fixture(autouse=True, scope="session")
def seed():
    seed = 42
    random.seed(seed)
    npr.seed(seed)
    torch.manual_seed(seed)
    return seed
