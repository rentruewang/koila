# Copyright (c) AIoWay Authors - All Rights Reserved

from sympy import Interval, S
from torch.nn import Dropout, Embedding, Linear, Transformer

from .previews import Preview
from .specs import SpecSet

__all__ = ["FACTORY"]

FACTORY: dict[str, Preview] = {}


FACTORY["LINEAR"] = Preview(
    module_class=Linear,
    limits=SpecSet.from_specs(in_features=S.Naturals, out_features=S.Naturals),
)

FACTORY["TRANSFORMER"] = Preview(
    module_class=Transformer,
    limits=SpecSet.from_specs(
        d_model=S.Naturals,
        nhead=S.Naturals,
        num_encoder_layers=S.Naturals,
        num_decoder_layers=S.Naturals,
        dim_feedforward=S.Naturals,
    ),
)

FACTORY["DROPOUT"] = Preview(
    module_class=Dropout,
    limits=SpecSet.from_specs(p=Interval(0, 1)),
)

FACTORY["EMBEDDING"] = Preview(
    module_class=Embedding,
    limits=SpecSet.from_specs(num_embeddings=S.Naturals, embedding_dim=S.Naturals),
)
