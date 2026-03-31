# Copyright (c) AIoWay Authors - All Rights Reserved

"The operators in relational algebra."

import typing
from collections import abc as cabc

import tensordict as td

from aioway._signs import Signature
from aioway._tracking import logging
from aioway._typing import SeqKeysView, SetKeysView

from . import _common
from .exprs import TensorDictExpr

__all__ = ["SelectTensorDictExpr", "RenameTensorDictExpr", "ZipTensorDictExpr"]


LOGGER = logging.get_logger(__name__)


@_common.expr_dcls
class SelectTensorDictExpr(TensorDictExpr):
    source: TensorDictExpr

    columns: cabc.Sequence[str]

    def keys(self) -> cabc.KeysView[str]:
        raise NotImplementedError

    @typing.override
    def _compute(self) -> td.TensorDict:
        pulled = self.source.compute()

        with _common.TRACKER(
            name="select",
            signature=Signature(td.TensorDict, td.TensorDict),
        ):
            return pulled.select(*self.columns)


@_common.expr_dcls
class RenameTensorDictExpr(TensorDictExpr):
    source: TensorDictExpr

    renames: dict[str, str]

    @typing.override
    def keys(self) -> cabc.KeysView[str]:
        return SeqKeysView([self.renames.get(key, key) for key in self.keys()])

    @typing.override
    def _compute(self) -> td.TensorDict:
        td = self.source.compute()
        with _common.TRACKER(
            name="rename",
            signature=Signature(td.TensorDict, td.TensorDict),
        ):
            return _rename(td, **self.renames)


@_common.expr_dcls
class ZipTensorDictExpr(TensorDictExpr):
    left: TensorDictExpr
    right: TensorDictExpr

    @typing.override
    def keys(self) -> cabc.KeysView[str]:
        return SetKeysView({*self.left.keys(), *self.right.keys()})

    @typing.override
    def _compute(self) -> td.TensorDict:
        left = self.left.compute()
        right = self.right.compute()

        with _common.TRACKER(
            name="zip",
            signature=Signature(td.TensorDict, td.TensorDict, td.TensorDict),
        ):
            return td.merge_tensordicts(left, right)


def _rename(td: td.TensorDict, **names: str) -> td.TensorDict:
    """
    Rename the columns of the current `Block`.
    """

    LOGGER.debug("Renamed called with names=%s", names)
    return td.TensorDict(
        {names.get(key, key): val for key, val in td.items()},
        batch_size=td.batch_size,
        device=td.device,
    )
