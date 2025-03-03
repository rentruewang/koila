# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from tensordict import TensorDict

from .frames import Frame

__all__ = ["TensorDictFrame"]


@dcls.dataclass(frozen=True)
class TensorDictFrame(Frame):
    """
    ``TensorDictFrame`` is a ``Frame`` backed by a ``TensorDict``.
    This means that a ``TensorDictFrame`` is entirely in-memory.
    """

    data: TensorDict
    """
    Underlying data of ``BlockFrame``.
    """

    def __len__(self) -> int:
        return len(self.data)

    def __getitem_tensordict(self, idx):
        return self.data[idx]

    _rows_int = _rows_slice = _rows_list = __getitem_tensordict
