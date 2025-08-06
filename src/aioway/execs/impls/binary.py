# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from abc import ABC

from aioway.execs.execs import Execution
from aioway.nodes import BinaryNode

__all__ = ["BinaryExec"]


# TODO
@dcls.dataclass
class BinaryExec(Execution, BinaryNode, ABC):
    left: Execution
    """
    The LHS of the operator.
    """

    right: Execution
    """
    The RHS of the operator.
    """
