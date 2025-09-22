# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
from abc import ABC

from torch import Tensor

from aioway._errors import AiowayError

__all__ = ["DistLoss"]


class LossFunc(ABC):
    """
    The loss used in supervisied learning,
    computes the distances between the input and the target.
    """

    def __call__(self, input: Tensor, target: Tensor, /) -> Tensor:
        result = self._compute(input, target)

        if result.size() != ():
            raise DistScalarError(f"The result must be a scalar. Got: {result.size()}")

        return result

    @abc.abstractmethod
    def _compute(self, input: Tensor, target: Tensor, /) -> Tensor:
        """
        The function subclasses must override to compute the distance.
        Result must be a scalar.

        Note that input and target does not neccessarily need to be of the same shape.
        """

        ...


class DistScalarError(AiowayError, ValueError): ...
