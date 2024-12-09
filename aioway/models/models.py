# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from torch.nn import Module

from aioway.blocks import TensorBlock

__all__ = ["Model"]


@dcls.dataclass(frozen=True)
class Model(Module):
    module: Module
    """
    The underlying module of the ``Model`` class.
    """

    device: str

    def forward(self, block: TensorBlock) -> TensorBlock:
        """
        The forward function of the model class.

        Todo:
            Should use ``inspect.signature`` to ensure that
            the wrapped module has a valid signature as described by ``Einsum``.

        Returns:
            The result computed by the underlying module.
        """

        return self.module(block)
