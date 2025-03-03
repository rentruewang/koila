# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

import tensordict
from tensordict import TensorDict

from aioway.errors import AiowayError

__all__ = ["ZipExec"]


@dcls.dataclass(frozen=True)
class ZipExec:
    def __call__(self, left: TensorDict, right: TensorDict) -> TensorDict:
        if same := set(left.keys()) & set(right.keys()):
            raise ZipMismatchError(
                f"Cannot concatenate blocks with the same keys: {same}"
            )

        if left.batch_size != right.batch_size:
            raise ZipMismatchError(
                "Cannot concatenate blocks due to a batch_size mismatch: "
                f"{left.batch_size=} != {right.batch_size=}"
            )

        return tensordict.merge_tensordicts(left, right)


class ZipMismatchError(AiowayError, ValueError): ...
