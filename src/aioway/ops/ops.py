# Copyright (c) AIoWay Authors - All Rights Reserved

"The operators (stateless)."

import dataclasses as dcls
import logging
from collections.abc import Callable
from typing import Any

from .signs import Signature

__all__ = ["Op"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class Op[T: Callable]:
    """
    The type that is returned by the registry, contains both the callable and the signature.
    """

    name: str
    "The name of the operator"

    signature: Signature
    "The signature of the funciton."

    function: T
    "The callable that is registered."

    def __post_init__(self) -> None:
        LOGGER.debug(f"Operator {self!s} created.")

        if not callable(self.function):
            raise ValueError("The function is not callable.")

    def __str__(self):
        return f"{self.name}<{self.signature}>"

    def __call__(self, *args: Any) -> Any:
        if not self.signature.params.check(*args):
            return NotImplemented

        # Calling the function. Doesn't expect any errors.
        answer = self.function(*args)

        if not isinstance(answer, self.signature.result):
            raise AssertionError(
                f"Expected {self.signature.result}, got {type(answer)=}."
            )

        return answer

    @property
    def __func__(self) -> T:
        return self.function
