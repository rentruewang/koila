from __future__ import annotations

import dataclasses as dcls
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence


@dataclass(init=False)
class ArgsKwargs:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    args: Sequence[Any] = dcls.field(default_factory=tuple)
    kwargs: Dict[str, Any] = dcls.field(default_factory=dict)


@dataclass(init=False)
class Condition:
    function: Callable[..., bool]
    arguments: Sequence[ArgsKwargs] = dcls.field(default_factory=list)

    def __init__(
        self,
        function: Callable[..., bool],
        arguments: Sequence[ArgsKwargs | Sequence[Any] | Dict[str, Any]],
    ) -> None:
        self.function = function
        self.arguments = []

        for argument in arguments:
            if isinstance(argument, Sequence):
                argument = ArgsKwargs(*argument)

            if isinstance(argument, dict):
                assert all(isinstance(key, str) for key in argument.keys())
                argument = ArgsKwargs(**argument)

            self.arguments.append(argument)

    def check(self) -> None:
        for argument in self.arguments:
            assert self.function(*argument.args, **argument.kwargs), [
                argument.args,
                argument.kwargs,
            ]
