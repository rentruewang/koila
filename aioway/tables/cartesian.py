# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import itertools

from .execs import BinaryExec
from .tables import Table, TableVisitor

__all__ = ["CartesianTable"]


@dcls.dataclass(frozen=True)
class CartesianTable(Table):
    """
    ``JoinTable`` is a ``Table`` that depends on 2 ``Table``s.
    For example, products in relational algebra can be implemented using a ``JoinTable``.
    """

    executor: BinaryExec
    "The function that takes 2 ``Table``s as inputs."

    left: Table
    "The left ``Table`` operator."

    right: Table
    "The right ``Table`` operator."

    def __iter__(self):
        for l, r in itertools.product(self.left, self.right):
            yield self.executor(l, r)

    def accept[T](self, visitor: TableVisitor[T]) -> T:
        return visitor.cartesian(self)

    @property
    def sources(self) -> tuple[Table, Table]:
        return self.left, self.right
