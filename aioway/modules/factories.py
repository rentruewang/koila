# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from sympy import Interval, S
from torch.nn import Dropout, Embedding, Linear, Module, Transformer

from .specs import SpecSet

__all__ = ["ModuleFactory"]


@dcls.dataclass(frozen=True)
class ModuleFactory:
    module_class: type[Module]
    """
    The corresponding ``torch`` module.
    This would be used in combination with ``preview``
    for initializing a ``torch`` module with parameters.
    """

    limits: SpecSet
    """
    A preview object that gives you information
    about the signature of ``module_class``.
    """


LINEAR = ModuleFactory(
    module_class=Linear,
    limits=SpecSet.from_specs(in_features=S.Naturals, out_features=S.Naturals),
)

TRANSFORMER = ModuleFactory(
    module_class=Transformer,
    limits=SpecSet.from_specs(
        d_model=S.Naturals,
        nhead=S.Naturals,
        num_encoder_layers=S.Naturals,
        num_decoder_layers=S.Naturals,
        dim_feedforward=S.Naturals,
    ),
)

DROPOUT = ModuleFactory(
    module_class=Dropout,
    limits=SpecSet.from_specs(p=Interval(0, 1)),
)

EMBEDDING = ModuleFactory(
    module_class=Embedding,
    limits=SpecSet.from_specs(num_embeddings=S.Naturals, embedding_dim=S.Naturals),
)
