# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from abc import ABC
from collections.abc import Sequence
from typing import Self

from aioway.plans import Node, Rewriter
from aioway.previews import Info, Preview, Registry
from aioway.relalg import Relation, RelationVisitor
from aioway.schemas import TableSchema
from aioway.tracers import Tracer


@dcls.dataclass(frozen=True)
class PreviewNode(Node["PreviewNode"]):
    """
    The node that holds a ``Preview`` as well as the traceback information.

    Todo:
        Rename this as the name isn't very descriptive.
    """

    preview: Preview

    inputs: Sequence[Self]

    def __call__(self) -> Info:
        return self.preview.compute(*(ipt() for ipt in self.inputs))

    @property
    def source_previews(self) -> Sequence[Preview]:
        return [ipt.preview for ipt in self.inputs]

    @property
    def sources(self) -> Sequence[Self]:
        return self.inputs

    @property
    def schema(self) -> TableSchema:
        raise NotImplementedError


@dcls.dataclass(frozen=True)
class Compiler(Rewriter):
    registry: Registry

    def __call__(self, source: Tracer) -> PreviewNode:
        return _GreedyCompilerByRelation(self).visit(source.relation)


@dcls.dataclass(frozen=True)
class RelationCompiler(RelationVisitor[Tracer, PreviewNode], ABC):
    compiler: Compiler

    @property
    def registry(self) -> Registry:
        return self.compiler.registry


@dcls.dataclass(frozen=True)
class _GreedyCompilerByRelation(RelationCompiler):
    def _leaf(self, relation: Relation[Tracer], /) -> PreviewNode:
        filtered = self.registry.filter_by_relation(type(relation))
        return PreviewNode(filtered[0], [])

    def _unary(self, relation: Relation[Tracer], /) -> PreviewNode:
        filtered = self.registry.filter_by_relation(type(relation))
        [source] = relation.sources
        return PreviewNode(filtered[0], [self.compiler(source)])

    def _binary(self, relation: Relation[Tracer], /) -> PreviewNode:
        filtered = self.registry.filter_by_relation(type(relation))
        left, right = relation.sources
        return PreviewNode(filtered[0], [self.compiler(left), self.compiler(right)])

    base = _leaf
    project = rename = select = transform = view = _unary
    concat = product = union = _binary
