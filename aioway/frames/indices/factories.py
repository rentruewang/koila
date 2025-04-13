# Copyright (c) RenChu Wang - All Rights Reserved

__all__ = ["index_factory"]

from typing import Any

from aioway.errors import AiowayError

from .faiss import FaissIndex
from .indices import Index, IndexContext
from .lexsort import LexsortIndex
from .ops import *


def index_factory(key: str | type[IndexOp], ctx: IndexContext, **kwargs: Any) -> Index:
    """
    todo))
        Currently ``index_factory`` would, based on what type of ``IndexOp`` is given,
        try to construct ``FaissIndex`` or ``LexsortIndex``.

        To scale this up, we would need a way to automatically handle multiple types of indices,
        much like how we handle machine learning algorithms and models.
    """

    if key in ["FAISS", IndexAnn]:
        return FaissIndex.create(ctx=ctx, **kwargs)

    if key in ["LEXSORT", IndexEq, IndexNe, IndexGe, IndexGt, IndexLe, IndexLt]:
        return LexsortIndex.create(ctx=ctx, **kwargs)

    raise IndexFactoryKeyError(
        "Only supports `FAISS` or `LEXSORT`, or one of the index operators currently. "
        f"Got key: {key}."
    )


class IndexFactoryKeyError(AiowayError, KeyError): ...
