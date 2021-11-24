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
class Caller:
    func: Callable[..., Any]
    arguments: Sequence[ArgsKwargs] = dcls.field(default_factory=list)

    def __init__(
        self,
        func: Callable[..., Any],
        arguments: Sequence[ArgsKwargs | Sequence[Any] | Dict[str, Any]],
    ) -> None:
        self.func = func
        self.arguments = []

        for argument in arguments:
            if isinstance(argument, Sequence):
                argument = ArgsKwargs(*argument)

            if isinstance(argument, dict):
                assert all(isinstance(key, str) for key in argument.keys())
                argument = ArgsKwargs(**argument)

            self.arguments.append(argument)

    def call(self) -> None:
        for argument in self.arguments:
            self.func(*argument.args, **argument.kwargs)


def call(
    func: Callable[..., Any],
    arguments: Sequence[ArgsKwargs | Sequence[Any] | Dict[str, Any]],
) -> None:
    Caller(func, arguments=arguments).call()


def is_notimplemented(func: Callable[[], Any]) -> bool:
    try:
        func()
        return False
    except:
        return True
