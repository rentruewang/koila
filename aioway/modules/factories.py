# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from torch.nn import Module

from .specs import SpecSet

__all__ = ["ModuleFactory"]


@dcls.dataclass(frozen=True)
class ModuleFactory:
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
