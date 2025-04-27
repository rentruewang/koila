# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Callable, Sequence

from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.execs import Exec

__all__ = ["MapExec", "ModuleExec"]


# TODO Improve the initialization of this class.
@dcls.dataclass(frozen=True)
class MapExecBase(Exec, ABC):
    """
    ``MapExec`` converts the input data stream with a custom function.
    """

    exe: Exec
    """
    The input ``Frame`` to perform computation on.
    """

    output: AttrSet
    """
    Output schema of the ``MapFrame``.
    """

    @typing.override
    def __next__(self) -> Block:
        item = next(self.exe)
        result = self.__call_compute(item)
        result.require_attrs(self.output)
        return result

    @abc.abstractmethod
    def _compute(self, item: Block) -> Block:
        """
        The computation on the input frame.
        """

        ...

    def __call_compute(self, item: Block) -> Block:
        """
        The computation on the input frame.
        """

        # Dynamically call subclasses' `_compute` method.
        result = self._compute(item)
        if not isinstance(result, Block):
            raise ModuleExecError(f"Output of {self} should be `Block`.")
        return result

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.output

    @property
    @typing.override
    def children(self) -> tuple[Exec]:
        return (self.exe,)


# NOTE
# Since `ModuleExec` is simply more powerful,
# as `torch` `Module`s are simply functions,
# with the only constraint being that they need to be `Module`s.
# Do I remove this completely?
@dcls.dataclass(frozen=True)
class MapExec(MapExecBase, key="MAP"):
    _: dcls.KW_ONLY
    """
    Made some changes to data ordering. This is the safest.
    """

    compute: Callable[[Block], Block]
    """
    The computation on the input frame.
    """

    @typing.override
    def _compute(self, item: Block) -> Block:
        return self.compute(item)


# TODO Extract information from previews directly.
@dcls.dataclass(frozen=True)
class ModuleExec(MapExecBase, key="MODULE"):
    """
    ``TensorDictModuleExec`` is an ``Exec`` that wraps a ``TensorDictModule``,
    and executes it on the input data.
    It is used to execute the module on the input data,
    and return the result as a ``Block``.
    """

    module: TensorDictModule
    """
    The module to be executed on the input data.
    """

    @typing.override
    def _compute(self, item: Block) -> Block:
        result = self.module(item.data)
        if not isinstance(result, Block):
            raise ModuleExecError(f"Output of {self.module=} should be `Block`.")
        return result

    @classmethod
    def wrap(
        cls, exe: Exec, function: Callable, in_keys: Sequence[str], output: AttrSet
    ):
        """
        Wrap a function into a ``ModuleExec``.
        The function must take a ``TensorDict`` as input,
        and return a ``TensorDict`` as output.
        """

        def wrapper(batch: TensorDict) -> TensorDict:
            result = function(batch.data)
            if not isinstance(result, TensorDict):
                raise ModuleExecError(
                    f"Function {function} returned a non-tensordict result: {result}"
                )
            return result

        module = TensorDictModule(
            module=wrapper, in_keys=in_keys, out_keys=output.columns
        )

        return ModuleExec(exe=exe, module=module, output=output)


class MapTypeError(AiowayError, TypeError): ...


class ModuleExecError(AiowayError, TypeError): ...
