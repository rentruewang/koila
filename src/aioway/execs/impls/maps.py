# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Callable, Iterator, Sequence

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.nn import Parameter

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.execs import Execution

from .unary import UnaryExec

__all__ = ["MapExec", "ModuleExec"]


@dcls.dataclass
class MapExecBase(UnaryExec, ABC):
    """
    ``MapExec`` converts the input data stream with a custom function.
    """

    output: AttrSet
    """
    Output schema of the ``MapFrame``.
    """

    @typing.override
    def __next__(self) -> Block:
        item = next(self._simple_iterator)
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


# TODO
@dcls.dataclass
class MapExec(MapExecBase, key="MAP"):
    """
    ``MapExec`` is an ``Exec`` that applies a function to the input data,
    and returns the result as a ``Block``.

    Note:
        Since ``MapExec`` is simply more powerful,
        as ``torch`` ``Module``s are simply functions,
        with the only constraint being that they need to be ``Module``s.
        Do I remove this completely?
    """

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


# TODO
@dcls.dataclass
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

    def __post_init__(self) -> None:
        if not isinstance(self.module, TensorDictModule):
            raise MapTypeError(f"Module {self.module} is not a `TensorDictModule`.")

    @typing.override
    def _compute(self, item: Block) -> Block:
        result = self.module(item.data)
        if not isinstance(result, Block):
            raise ModuleExecError(f"Output of {self.module=} should be `Block`.")
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Zero the gradients of the module.
        """

        self.module.zero_grad(set_to_none=set_to_none)

    def parameters(self) -> Iterator[Parameter]:
        """
        Returns the parameters of the module.
        """

        return self.module.parameters()

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        """
        Returns the named parameters of the module.
        """

        return self.module.named_parameters()

    @classmethod
    def wrap(
        cls, exe: Execution, function: Callable, in_keys: Sequence[str], output: AttrSet
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

        return ModuleExec(child=exe, module=module, output=output)


class MapTypeError(AiowayError, TypeError): ...


class ModuleExecError(AiowayError, TypeError): ...
