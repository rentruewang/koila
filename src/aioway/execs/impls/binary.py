# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from abc import ABC

from aioway.execs.execs import Exec
from aioway.nodes import BinaryNode

__all__ = ["BinaryExec"]


@dcls.dataclass
class BinaryExec(Exec, BinaryNode, ABC):
    left: Exec
    """
    The LHS of the operator.
    """

    right: Exec
    """
    The RHS of the operator.
    """
