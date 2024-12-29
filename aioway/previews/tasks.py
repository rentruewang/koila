# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections.abc import Sequence
from typing import NamedTuple

from aioway.plans import Dag
from aioway.schemas import TableSchema

from .einsums import Einsum


class Task(NamedTuple):
    """
    The description of a task, with a list of defined inputs and a list of defined outputs.
    """

    input: tuple[TableSchema, ...]
    """
    The schema of the input
    """

    output: tuple[TableSchema, ...]
    """
    The schema of the output.
    """


@dcls.dataclass(frozen=True)
class Compiler(ABC):
    @abc.abstractmethod
    def __call__(self, task: Task, reg: Sequence[Einsum], /) -> Dag[Einsum]: ...
