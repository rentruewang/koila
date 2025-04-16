# Copyright (c) RenChu Wang - All Rights Reserved

__all__ = ["ModulePreview"]


import dataclasses as dcls
from collections.abc import Callable

from sympy import Interval, S
from torch.nn import Dropout, Embedding, Linear, Module, Transformer

from .limits import LimitSet


@dcls.dataclass(frozen=True)
class ModulePreview:
    module_class: type[Module]
    """
    The corresponding ``torch`` module.
    This would be used in combination with ``preview``
    for initializing a ``torch`` module with parameters.
    """

    limits: LimitSet
    """
    A preview object that gives you information
    about the signature of ``module_class``.
    """

    # FIXME
    #   Return -1 because I do not yet want to implement everything as of yet.
    num_params: Callable[[LimitSet], int] = lambda _: -1
    """
    Compute the number of parameters from the set of constraints.
    """


LINEAR = ModulePreview(
    module_class=Linear,
    limits=LimitSet.from_params(in_features=S.Naturals, out_features=S.Naturals),
    # There is also the bias term so +1.
    num_params=lambda ls: (ls["in_features"] + 1) * ls["out_features"],
)

TRANSFORMER = ModulePreview(
    module_class=Transformer,
    limits=LimitSet.from_params(
        d_model=S.Naturals,
        nhead=S.Naturals,
        num_encoder_layers=S.Naturals,
        num_decoder_layers=S.Naturals,
        dim_feedforward=S.Naturals,
    ),
)

DROPOUT = ModulePreview(
    module_class=Dropout,
    limits=LimitSet.from_params(p=Interval(0, 1)),
    num_params=lambda _: 0,
)

EMBEDDING = ModulePreview(
    module_class=Embedding,
    limits=LimitSet.from_params(num_embeddings=S.Naturals, embedding_dim=S.Naturals),
    num_params=lambda ls: ls["num_embeddings"] * ls["embedding_dim"],
)
