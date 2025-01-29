# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import operator
import typing
from abc import ABC
from collections.abc import Callable, Iterator
from typing import Protocol

from torch import Tensor

from aioway.blocks import TensordictBlock
from aioway.plans import Node
from aioway.relalg import BinaryExpr, Expr, LeafExpr, UnaryExpr

from .execs import BinaryExec, UnaryExec

if typing.TYPE_CHECKING:
    from .cartesian import CartesianTable
    from .linear import LinearTable
    from .maps import MapTable
    from .zips import ZipTable

__all__ = ["Table", "TableVisitor"]


@dcls.dataclass(frozen=True)
class Table(Node["Table"], ABC):
    """
    ``Table`` is the main physical abstraction of the project,
    not only capturing symbolic relationships like ``PlanNode`` does,
    but also stores iteration logic and computation between different nodes,
    as well as interacting with models and performing I/O operation.

    It is somewhat similar to RDD in spark,
    except the abstraction is not a partitioned list.

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
        Document difference between ``Table``, ``Block``, ``DataFrame``.

    Note:
        I was thinking whether or not ``Table`` handles too many things, however,
        I decided to proceed because this approach is adopted by most other libraries,
        including a few key libraries that ``aioway`` depends on,
        like ``torch.nn.Module`` acting differently on 1 node vs in distributed settings.
        Similarly, ``spark``'s ``RDD``s also handle data and computation this way, ``ibis`` as well.

    Todo:
        Reduce the number of member functions by using mixins.

    Todo:
        ``Table``s should possibly also have their own states.
    """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[TensordictBlock]: ...

    @abc.abstractmethod
    def accept[T](self, visitor: "TableVisitor[T]") -> T: ...

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

    def join(self, other: "Table", op: BinaryExec, /) -> "CartesianTable":
        """
        Join operation to compute joins between 2 ``Table``s.

        Args:
            other: The other table.
            op: A binary operator.

        Returns:
            A table that can be evaluated into data.
        """

        from .cartesian import CartesianTable

        return CartesianTable(op, self, other)

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

    def concat(self, other: "Table") -> "CartesianTable":
        from .cartesian import CartesianTable

        return CartesianTable(lambda left, right: left.concat(right), self, other)

    def union(self, other: "Table") -> "CartesianTable":
        from .cartesian import CartesianTable

        return CartesianTable(lambda left, right: left.union(right), self, other)


def _index_for_select(block: TensordictBlock, expr: Expr) -> Tensor:

    match expr:
        case LeafExpr(value):
            if isinstance(value, str) and value in block.schema():
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


class TableVisitor[T](Protocol):
    def visit(self, table: "Table", /) -> T:
        return table.accept(self)

    @abc.abstractmethod
    def cartesian(self, table: "CartesianTable", /) -> T: ...

    @abc.abstractmethod
    def linear(self, table: "LinearTable") -> T: ...

    @abc.abstractmethod
    def map(self, table: "MapTable", /) -> T: ...

    @abc.abstractmethod
    def zip(self, table: "ZipTable", /) -> T: ...
