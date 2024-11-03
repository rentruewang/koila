# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import typing
from collections.abc import Sequence
from typing import Protocol, TypeVar

from aioway.logics.dtypes import Schema

from .nodes import PlanNode

if typing.TYPE_CHECKING:
    from .bases import BaseRelation
    from .products import ConcatRelation, ProductRelation
    from .projections import ProjectionRelation
    from .renames import RenameRelation
    from .selections import SelectionRelation
    from .transforms import TransformRelation
    from .unions import UnionRelation
    from .views import ViewRelation

_T = TypeVar("_T", covariant=True)


class Relation(Protocol):
    """
    The base class for all relations in the relational algebra.
    """

    class Visitor(Protocol[_T]):
        """
        The visitor for ``Relation``s.

        Currently supports the following types:

        - ``Base``
        - ``CartesianProduct``
        - ``Projection``
        - ``Rename``
        - ``Selection``
        - ``Transform``
        - ``Union``
        - ``View``
        """

        def __call__(self, relation: "Relation", /) -> _T:
            return relation.accept(self)

        @abc.abstractmethod
        def base(self, relation: "BaseRelation", /) -> _T: ...

        @abc.abstractmethod
        def concat(self, relation: "ConcatRelation") -> _T: ...

        @abc.abstractmethod
        def product(self, relation: "ProductRelation", /) -> _T: ...

        @abc.abstractmethod
        def project(self, relation: "ProjectionRelation", /) -> _T: ...

        @abc.abstractmethod
        def rename(self, relation: "RenameRelation", /) -> _T: ...

        @abc.abstractmethod
        def select(self, relation: "SelectionRelation", /) -> _T: ...

        @abc.abstractmethod
        def transform(self, relation: "TransformRelation", /) -> _T: ...

        @abc.abstractmethod
        def union(self, relation: "UnionRelation", /) -> _T: ...

        @abc.abstractmethod
        def view(self, relation: "ViewRelation", /) -> _T: ...

    @abc.abstractmethod
    def accept(self, visitor: "Relation.Visitor[_T]", /) -> _T: ...

    @property
    @abc.abstractmethod
    def sources(self) -> Sequence[PlanNode]:
        """
        The previous tables the relation comes from.
        The sequence length is at least 1 ``Node``.

        Returns:
            The previous tables.
        """

        ...

    @property
    @abc.abstractmethod
    def schema(self) -> Schema:
        """
        The schema of the current relation.

        Depending on the relation,
        the schema can either be derived from the relation node's input,
        or directly inferred from the input data.
        """

        ...
