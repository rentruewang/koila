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


@dataclass
class Condition:
    function: Callable[..., bool]
    arguments: Sequence[ArgsKwargs] = dcls.field(default_factory=list)

    def check(self) -> None:
        for argument in self.arguments:
            assert self.function(*argument.args, **argument.kwargs), [
                argument.args,
                argument.kwargs,
            ]
