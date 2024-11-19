# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import typing
from collections.abc import Sequence
from typing import Protocol, TypeVar

from aioway.schemas import TableSchema

from .nodes import RelNode

if typing.TYPE_CHECKING:
    from .bases import BaseRelation
    from .products import ConcatRelation, ProductRelation
    from .projections import ProjectionRelation
    from .renames import RenameRelation
    from .selections import SelectionRelation
    from .transforms import TransformRelation
    from .unions import UnionRelation
    from .views import ViewRelation

__all__ = ["Relation", "RelationVisitor"]

T = TypeVar("T", covariant=True)
P = TypeVar("P", bound=RelNode)


class Relation(Protocol[P]):
    """
    The base class for all relations in the relational algebra.
    """

    @abc.abstractmethod
    def accept(self, visitor: "RelationVisitor[P, T]", /) -> T: ...

    @property
    @abc.abstractmethod
    def sources(self) -> Sequence[P]:
        """
        The previous tables the relation comes from.
        The sequence length is at least 1 ``Node``.

        Returns:
            The previous tables.
        """

        ...

    @property
    @abc.abstractmethod
    def schema(self) -> TableSchema:
        """
        The schema of the current relation.

        Depending on the relation,
        the schema can either be derived from the relation node's input,
        or directly inferred from the input data.
        """

        ...


class RelationVisitor(Protocol[P, T]):
    """
    The visitor for ``Relation``s.

    Currently supports the following types:

    - ``Base``
    - ``Concat``
    - ``Product``
    - ``Projection``
    - ``Rename``
    - ``Selection``
    - ``Transform``
    - ``Union``
    - ``View``
    """

    def visit(self, relation: "Relation[P]", /) -> T:
        return relation.accept(self)

    @abc.abstractmethod
    def base(self, relation: "BaseRelation[P]", /) -> T: ...

    @abc.abstractmethod
    def concat(self, relation: "ConcatRelation[P]", /) -> T: ...

    @abc.abstractmethod
    def product(self, relation: "ProductRelation[P]", /) -> T: ...

    @abc.abstractmethod
    def project(self, relation: "ProjectionRelation[P]", /) -> T: ...

    @abc.abstractmethod
    def rename(self, relation: "RenameRelation[P]", /) -> T: ...

    @abc.abstractmethod
    def select(self, relation: "SelectionRelation[P]", /) -> T: ...

    @abc.abstractmethod
    def transform(self, relation: "TransformRelation[P]", /) -> T: ...

    @abc.abstractmethod
    def union(self, relation: "UnionRelation[P]", /) -> T: ...

    @abc.abstractmethod
    def view(self, relation: "ViewRelation[P]", /) -> T: ...
