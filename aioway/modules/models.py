# Copyright (c) RenChu Wang - All Rights Reserved

import tensordict
from tensordict.nn import TensorDictModule

from aioway.blocks import Block

__all__ = ["Model"]


# TODO
#   Should use `inspect.signature` to ensure that
#   the wrapped module has a valid signature as described by `Einsum`.
@tensordict.tensorclass(frozen=True)
class Model:
    module: TensorDictModule
    """
    The underlying module of the ``Model`` class.
    """

    device: str

    def forward(self, block: Block) -> Block:
        """
        The forward function of the model class.


        Returns:
            The result computed by the underlying module.
        """

        raise NotImplementedError
