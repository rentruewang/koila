# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
from collections.abc import Iterator
from dataclasses import KW_ONLY as KwOnly
from typing import Any, Self

from aioway.registries import ClassRegistry, Registry, types

from .execs import Exec
from .ops import Op
from .pollers import Poller

__all__ = ["ExecInit"]


@dcls.dataclass(frozen=True)
class ExecInit:
    """
    ``ExecInit`` is an initializer for ``Exec``,
    s.t. it can be ``__call__``-ed and construct an ``Exec``.
    """

    _: KwOnly

    poller: type[Poller]
    """
    The class for the ``Poller`` to use.
    """

    op: type[Op]
    """
    The type of the ``Op`` to use.
    """

    def __call__(self, *inputs: Exec, **op_opts: Any) -> Exec:
        """
        Construct an ``Exec``,
        based on the ``inputs`` and the options for the current ``Op`` ``op_opts``.

        Args:
            *inputs: The inputs to use. Must be the same count as ``Poller.ARGC``.
            **op_opts: The options for ``Op``.

        Returns:
            Constructed ``Exec`` instance.
        """

        return Exec(
            poller_type=self.poller,
            op_type=self.op,
            execs=inputs,
            op_opts=op_opts,
        )

    @classmethod
    def from_keys(cls, poller: str, op: str):
        poller_reg: ClassRegistry[type[Poller]] = types.of(Poller)
        op_reg: ClassRegistry[type[Op]] = types.of(Op)
        return cls(poller=poller_reg[poller], op=op_reg[op])

    @classmethod
    @functools.cache
    def preset(cls) -> Registry[Self]:
        return Registry(
            {
                name: cls.from_keys(poller, op)
                for name, poller, op in _exec_bundle_keys()
            }
        )


def _exec_bundle_keys() -> Iterator[tuple[str, str, str]]:
    """
    The 3 tuple of key for `Executor`, `Poller`, `Op`.
    """

    # 0-ary.
    yield "FRAME", "NOOP_0", "FRAME_0"

    # 1-ary
    yield "FILTER_PRED", "PASS_1", "FUNC_FILTER_1"
    yield "FILTER_EXPR", "PASS_1", "EXPR_FILTER_1"
    yield "MAP", "PASS_1", "MAP_1"
    yield "MODULE", "PASS_1", "MODULE_1"
    yield "PROJECT", "PASS_1", "PROJECT_1"
    yield "RENAME", "PASS_1", "RENAME_1"

    # 2-ary
    yield "ZIP", "ZIP_2", "ZIP_2"
    yield "NESTED_LOOP", "NESTED_2", "MATCH_2"
