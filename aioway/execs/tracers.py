# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import inspect
import typing
from typing import NamedTuple, Self

from aioway import factories
from aioway.attrs import AttrSet
from aioway.errors import AiowayError
from aioway.plans import PlanNode

from .binary import BinaryExec
from .execs import Exec
from .nullary import NullaryExec
from .unary import UnaryExec

__all__ = ["ExecTracer"]


@typing.final
@dcls.dataclass(frozen=True)
class ExecTracer(PlanNode):
    """
    ``ExecTracer`` is a wrapper around ``Exec`` to track the execution of the plan,
    it acts as a convenience wrapper around ``Exec`` to provide a more user-friendly interface.
    """

    exe: Exec
    """
    The input ``Exec`` of the current ``ExecTracer``.
    """

    def map(self, op: str, *args, **kwargs) -> Self:
        init = InitExec(
            operator=op,
            exec_class=factories.of(Exec)[op],
            super_class=UnaryExec,
            identifier="unary",
        )
        exe = init(self.exe, *args, **kwargs)
        return dcls.replace(self, exe=exe)

    def join(self, other: Self, op: str, *args, **kwargs) -> Self:
        init = InitExec(
            operator=op,
            exec_class=factories.of(Exec)[op],
            super_class=BinaryExec,
            identifier="binary",
        )
        exe = init(self.exe, other.exe, *args, **kwargs)
        return dcls.replace(self, exe=exe)

    @classmethod
    def create(cls, op: str, *args, **kwargs) -> Self:
        init = InitExec(
            operator=op,
            exec_class=factories.of(Exec)[op],
            super_class=NullaryExec,
            identifier="nullary",
        )
        exe = init(*args, **kwargs)
        return cls(exe=exe)

    @property
    def attrs(self) -> AttrSet:
        """
        The attributes of the current ``ExecTracer``.
        """

        return self.exe.attrs

    @property
    @typing.override
    def children(self) -> tuple[Exec, ...]:
        return self.exe.children


class InitExec(NamedTuple):
    """
    ``InitExec`` is a wrapper around ``Exec``,
    to provide code sharing for the ``ExecTracer`` class.
    """

    operator: str
    """
    The operator of the current ``ExecTracer``.
    """

    exec_class: type[Exec]
    """
    The class of the current ``ExecTracer``.
    """

    super_class: type
    """
    The super class of the current ``ExecTracer``.
    """

    identifier: str
    """
    The identifier of the current ``ExecTracer``.
    """

    def __call__(self, *args, **kwargs) -> Exec:
        self.check_subclass()
        return self.instantiate(*args, **kwargs)

    def check_subclass(self) -> None:
        """
        Check if the given class is a subclass of the base
        class, and if not, raise an error.
        """

        if issubclass(self.exec_class, self.super_class):
            return

        raise ExecTracerOpError(
            f"ExecTracer only supports {self.identifier} operations, but {self.operator} is not a {self.identifier} operation."
        )

    def instantiate(self, *args, **kwargs) -> Exec:
        """
        Instantiate the given class with the given arguments.
        """

        try:
            exe = self.exec_class(*args, **kwargs)
        except TypeError:
            raise ExecTracerInitError(
                f"Failed to instantiate {self.operator} with {args=}, {kwargs=}. "
                f"Signature of {self.exec_class} is {inspect.signature(self.exec_class)}. "
            )
        return exe


class ExecTracerOpError(AiowayError, ValueError): ...


class ExecTracerInitError(AiowayError, TypeError): ...
