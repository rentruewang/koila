# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import operator
import typing
from abc import ABC
from collections.abc import Callable
from typing import Protocol, TypeVar

from torch import Tensor

from aioway.blocks import BinaryExec, Block, UnaryExec
from aioway.plans import Node
from aioway.relalg import BinaryExpr, Expr, LeafExpr, UnaryExpr

if typing.TYPE_CHECKING:
    from .joins import JoinTable
    from .maps import MapTable
    from .sources import SourceTable

__all__ = ["Table", "TableVisitor"]

T = TypeVar("T", covariant=True)


@dcls.dataclass(frozen=True)
class Table(Node["Table"], ABC):
    """
    ``Table`` is the main physical abstraction of the project,
    not only capturing symbolic relationships like ``PlanNode`` does,
    but also stores iteration logic and computation between different nodes.

    It is somewhat similar to RDD in spark,
    except the abstraction is not a partitioned list, but rather an iterable.

    This is due to the format being most similar to most machine learning workflows,
    where most models follow a batched pattern because of the size of the dataset.
    The same can also be said for streaming as well.

    This abstraction is also extensible to have random access and distributed partition,
    with supports of indices and partitions, due to only requiring iteration.

    Todo:
        ``Table`` only supports synchronous execution, which is to say,
        a ``Table`` can only be computed once all its dependencies are all executed.

        Think about how to expand this into streaming contexts.

    Todo:
        ``Table``s should possibly also have their own states.
        This is important to represent for example
    """

    @abc.abstractmethod
    def __call__(self) -> Block: ...

    @abc.abstractmethod
    def accept(self, visitor: "TableVisitor[T]") -> T: ...

    def map(self, op: UnaryExec, /) -> "MapTable":
        """
        Map operation to transform current ``Table`` into another ``Table``.

        Args:
            op: A unary function.

        Returns:
            A table that can be evaluated into data.
        """

        from .maps import MapTable

        return MapTable(op, self)

    def join(self, other: "Table", op: BinaryExec, /) -> "JoinTable":
        """
        Join operation to compute joins between 2 ``Table``s.

        Args:
            other: The other table.
            op: A binary operator.

        Returns:
            A table that can be evaluated into data.
        """

        from .joins import JoinTable

        return JoinTable(op, self, other)

    def all(self) -> bool:
        return self.all()

    def any(self) -> bool:
        return self.all()

    def select(self, expr: Expr):
        from .maps import MapTable

        return MapTable(lambda blk: blk[_index_for_select(blk, expr).tolist()], self)

    def project(self, *keys: str) -> "MapTable":
        from .maps import MapTable

        return MapTable(lambda block: block.project(*keys), self)

    def rename(self, **renames: str) -> "MapTable":
        from .maps import MapTable

        return MapTable(lambda block: block.rename(*renames), self)

    def concat(self, other: "Table") -> "JoinTable":
        from .joins import JoinTable

        return JoinTable(lambda left, right: left.concat(right), self, other)

    def union(self, other: "Table") -> "JoinTable":
        from .joins import JoinTable

        return JoinTable(lambda left, right: left.union(right), self, other)


def _index_for_select(block: Block, expr: Expr) -> Tensor:

    match expr:
        case LeafExpr(value):
            if isinstance(value, str) and value in block.schema:
                return block.data[value]
            raise KeyError
        case UnaryExpr(op, operand):
            evaluated = _index_for_select(block, operand)

            unary: Callable[[Tensor], Tensor]
            match op:
                case "-":
                    unary = operator.neg
                case "~":
                    unary = operator.inv
                case _:
                    raise ValueError

            return unary(evaluated)

        case BinaryExpr(op, left, right):
            eval_left = _index_for_select(block, left)
            eval_right = _index_for_select(block, right)

            binary: Callable[[Tensor, Tensor], Tensor]
            match op:
                case "+":
                    binary = operator.add
                case "-":
                    binary = operator.sub
                case "*":
                    binary = operator.mul
                case ">":
                    binary = operator.gt
                case ">=":
                    binary = operator.ge
                case "<":
                    binary = operator.lt
                case "<=":
                    binary = operator.le
                case "=" | "==":
                    binary = operator.eq
                case "!=" | "<>":
                    binary = operator.ne
                case _:
                    raise ValueError

            return binary(eval_left, eval_right)
    raise ValueError


class TableVisitor(Protocol[T]):
    def visit(self, table: "Table", /) -> T:
        return table.accept(self)

    @abc.abstractmethod
    def source(self, table: "SourceTable") -> T: ...

    @abc.abstractmethod
    def map(self, table: "MapTable", /) -> T: ...

    @abc.abstractmethod
    def join(self, table: "JoinTable", /) -> T: ...
