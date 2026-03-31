# Copyright (c) AIoWay Authors - All Rights Reserved

"The operators in relational algebra."

import typing
from collections import abc as cabc

import tensordict as td

from aioway import _signs, _typing
from aioway._tracking import logging

from . import _common, exprs

__all__ = ["SelectTensorDictExpr", "RenameTensorDictExpr", "ZipTensorDictExpr"]


LOGGER = logging.get_logger(__name__)


@_common.expr_dcls
class SelectTensorDictExpr(exprs.TensorDictExpr):
    source: exprs.TensorDictExpr

    columns: cabc.Sequence[str]

    def keys(self) -> cabc.KeysView[str]:
        raise NotImplementedError

    @typing.override
    def _compute(self) -> td.TensorDict:
        pulled = self.source.compute()

        with _common.TRACKER(
            name="select",
            signature=_signs.Signature(td.TensorDict, td.TensorDict),
        ):
            return pulled.select(*self.columns)


@_common.expr_dcls
class RenameTensorDictExpr(exprs.TensorDictExpr):
    source: exprs.TensorDictExpr

    renames: dict[str, str]

    @typing.override
    def keys(self) -> cabc.KeysView[str]:
        return _typing.SeqKeysView([self.renames.get(key, key) for key in self.keys()])

    @typing.override
    def _compute(self) -> td.TensorDict:
        data = self.source.compute()
        with _common.TRACKER(
            name="rename",
            signature=_signs.Signature(td.TensorDict, td.TensorDict),
        ):
            return _rename(data, **self.renames)


@_common.expr_dcls
class ZipTensorDictExpr(exprs.TensorDictExpr):
    left: exprs.TensorDictExpr
    right: exprs.TensorDictExpr

    @typing.override
    def keys(self) -> cabc.KeysView[str]:
        return _typing.SetKeysView({*self.left.keys(), *self.right.keys()})

    @typing.override
    def _compute(self) -> td.TensorDict:
        left = self.left.compute()
        right = self.right.compute()

        with _common.TRACKER(
            name="zip",
            signature=_signs.Signature(td.TensorDict, td.TensorDict, td.TensorDict),
        ):
            return td.merge_tensordicts(left, right)


def _rename(data: td.TensorDict, **names: str) -> td.TensorDict:
    """
    Rename the columns of the current `Block`.
    """

    LOGGER.debug("Renamed called with names=%s", names)
    return td.TensorDict(
        {names.get(key, key): val for key, val in data.items()},
        batch_size=data.batch_size,
        device=data.device,
    )
