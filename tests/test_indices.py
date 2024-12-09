# Copyright (c) RenChu Wang - All Rights Reserved

import pytest
import torch
from torch import Tensor

from aioway.indices import FaissIndex, Index


@pytest.fixture
def data() -> Tensor:
    return torch.randn(101, 51)


@pytest.fixture
def index_type() -> str:
    return "Flat"


@pytest.fixture
def faiss_index(data) -> Index:
    return FaissIndex.index_factory()
