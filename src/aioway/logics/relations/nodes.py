# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import typing
from collections.abc import Sequence
from typing import Protocol

from aioway.logics.dtypes import Schema

if typing.TYPE_CHECKING:
    from .relations import Relation


@typing.runtime_checkable
class PlanNode(Protocol):
    """
    ``PlanNode`` is like a table in SQL,
    either computed from a ``Relation`` of tables (``CREATE VIEW``),
    where its values and data types depend on the previous tables,
    or as a concrete table (``CREATE TABLE``),
    where it is responsible for storing its own data types in a ``Base`` relation.

    It only captures logical relationships like ``Exec`` nodes or ``SparkPlan`` in spark.
    """

    @property
    @abc.abstractmethod
    def relation(self) -> "Relation":
        """
        The relation that derives from previous tables.
        """

    @property
    def schema(self) -> Schema:
        """
        The names of the columns in this plan node.

        Returns:
            A sequence of strings.
        """

        return self.relation.schema

    @property
    def sources(self) -> Sequence["PlanNode"]:
        """
        Get the dependencies of this current node.
        """

        return self.relation.sources
