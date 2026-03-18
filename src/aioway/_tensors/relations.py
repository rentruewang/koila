# Copyright (c) AIoWay Authors - All Rights Reserved

"The operators in relational algebra."

from collections.abc import KeysView, Sequence

import tensordict as td
from tensordict import TensorDict

from aioway import _logging
from aioway._typing import SeqKeysView, SetKeysView

from . import _common
from .exprs import TensorDictExpr

__all__ = ["SelectTensorDictExpr", "RenameTensorDictExpr", "ZipTensorDictExpr"]


LOGGER = _logging.get_logger(__name__)


@_common.expr_dcls
class SelectTensorDictExpr(TensorDictExpr):
    source: TensorDictExpr

    columns: Sequence[str]

    def keys(self) -> KeysView[str]:
        raise NotImplementedError

    def _compute(self) -> TensorDict:
        return self.source.compute().select(*self.columns)

    def _inputs(self):
        return (self.source,)


@_common.expr_dcls
class RenameTensorDictExpr(TensorDictExpr):
    source: TensorDictExpr

    renames: dict[str, str]

    def keys(self) -> KeysView[str]:
        return SeqKeysView([self.renames.get(key, key) for key in self.keys()])

    def _compute(self) -> TensorDict:
        td = self.source.compute()
        return _rename(td, **self.renames)

    def _inputs(self):
        return (self.source,)


@_common.expr_dcls
class ZipTensorDictExpr(TensorDictExpr):
    left: TensorDictExpr
    right: TensorDictExpr

    def keys(self) -> KeysView[str]:
        return SetKeysView({*self.left.keys(), *self.right.keys()})

    def _compute(self) -> TensorDict:
        left = self.left.compute()
        right = self.right.compute()
        return td.merge_tensordicts(left, right)

    def _inputs(self):
        return self.left, self.right


def _rename(td: TensorDict, **names: str) -> TensorDict:
    """
    Rename the columns of the current `Block`.
    """

    LOGGER.debug("Renamed called with names=%s", names)
    return TensorDict(
        {names.get(key, key): val for key, val in td.items()},
        batch_size=td.batch_size,
        device=td.device,
    )
