# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Sequence

import pandas as pd

from aioway.backend import SourceTable, Table
from aioway.logics import (
    BaseRelation,
    ConcatRelation,
    Expr,
    Product,
    ProductRelation,
    ProjectionRelation,
    Relation,
    RelationVisitor,
    RenameRelation,
    Schema,
    SelectionRelation,
    TransformRelation,
    UnionRelation,
    ViewRelation,
)


@dcls.dataclass(frozen=True)
class Tracer:
    """
    The main bridge between logical world and physical world.

    ``Tracer``s act as an intermediate representation,
    while tracing the logical graph and convert into a physical plan.
    """

    relation: Relation["Tracer"]

    @property
    def schema(self) -> Schema:
        return self.relation.schema

    @property
    def sources(self) -> Sequence["Tracer"]:
        return self.relation.sources

    def concat(self, other: "Tracer") -> "Tracer":
        return type(self)(ConcatRelation(self, other))

    def product(
        self, other: "Tracer", keys: tuple[str, str], prod: Product
    ) -> "Tracer":
        return type(self)(ProductRelation(self, other, keys=keys, prod=prod))

    def project(self, *columns: str) -> "Tracer":
        return type(self)(ProjectionRelation(self, columns))

    def rename(self, **cols: str) -> "Tracer":
        return type(self)(RenameRelation(self, cols))

    def select(self, expr: Expr) -> "Tracer":
        return type(self)(SelectionRelation(self, expr))

    def transform(self, to: Schema) -> "Tracer":
        return type(self)(TransformRelation(self, to))

    def union(self, other: "Tracer") -> "Tracer":
        return type(self)(UnionRelation(self, other))

    def view(self) -> "Tracer":
        return type(self)(ViewRelation(self))

    @classmethod
    def source(cls, schema: Schema, filename: str) -> "Tracer":
        return cls(BaseRelation(schema, filename))

    def table(self) -> Table:
        return TracerCompiler()(self)


class TracerCompiler(RelationVisitor[Tracer, Table]):
    def __call__(self, tracer: Tracer) -> Table:
        return self.visit(tracer.relation)

    def base(self, relation: BaseRelation[Tracer]) -> Table:
        """
        Compile the base ``Relation`` (the input relation) by reading from a file.

        Args:
            relation: The base relation that specified a file.

        Returns:
            A ``Table`` instance.

        Todo:
            This doesn't look very elegant, and requires loading csv into memory.
            Make changes s.t. none of this is needed.
        """

        df = pd.read_csv(relation.filename)
        return SourceTable.from_pandas(df)

    def concat(self, relation: ConcatRelation[Tracer]) -> Table:
        left = self(relation.left)
        right = self(relation.right)
        return left.concat(right)

    def product(self, relation: ProductRelation[Tracer]) -> Table:
        """
        Todo:
            Add in join operations.
        """

        raise NotImplementedError("Join operations are not implemented yet.")

    def project(self, relation: ProjectionRelation[Tracer]) -> Table:
        prev = self(relation.prev)
        return prev.project(*relation.columns)

    def rename(self, relation: RenameRelation[Tracer]) -> Table:
        prev = self(relation.prev)
        return prev.rename(**relation.cols)

    def select(self, relation: SelectionRelation[Tracer]) -> Table:
        prev = self(relation.prev)
        return prev.select(relation.condition)

    def transform(self, relation: TransformRelation[Tracer]) -> Table:
        """
        Todo:
            Insert models somewhere.
        """

        raise NotImplementedError("Transform types are not implemented yet.")

    def union(self, relation: UnionRelation[Tracer]) -> Table:
        top = self(relation.top)
        down = self(relation.down)
        return top.union(down)

    def view(self, relation: ViewRelation[Tracer]) -> Table:
        return self(relation.prev)
