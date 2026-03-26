# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import Tensor
from torch.nn import Embedding as TorchEmbedding

from aioway.attrs import Attr
from aioway.nn import Embedding


@pytest.fixture
def emb():
    return Embedding(num_embeddings=3, embedding_dim=5)


@pytest.fixture
def nn_emb(emb: Embedding):
    return emb.MODULE_TYPE(num_embeddings=3, embedding_dim=5)


@pytest.fixture
def ok_input():
    return torch.randn(11, 3).int()


@pytest.fixture
def bad_input():
    return torch.randn(11, 7)


def test_emb_forward(ok_input: Tensor, emb: Embedding, nn_emb: TorchEmbedding):
    assert emb.forward(ok_input).shape == nn_emb(ok_input).shape
    assert emb.forward(ok_input).dtype == nn_emb(ok_input).dtype


def test_emb_preview(ok_input: Tensor, emb: Embedding, nn_emb: TorchEmbedding):
    assert emb.preview(Attr.from_tensor(ok_input)).shape == nn_emb(ok_input).shape
    assert emb.preview(Attr.from_tensor(ok_input)).dtype == nn_emb(ok_input).dtype


def test_emb_preview_fail(bad_input: Tensor, emb: Embedding):
    assert emb.preview(Attr.from_tensor(bad_input)) is NotImplemented
