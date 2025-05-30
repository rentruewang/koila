# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

from torch.nn import Module

from .specs import SpecSet

__all__ = ["Preview"]


@dcls.dataclass(frozen=True)
class Preview:
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

    # contract: EinsumAttr
    # """
    # The contract for which the module would satisfy.
    # """
