# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC

from torch import Tensor

from aioway.errors import AiowayError

__all__ = ["DistLoss"]


# TODO Integrate this with `Attr` / `AttrSet`
# TODO
#   How do I give the constraints on the inputs?
#   Perhaps we need to add more metadata on the outputs range.
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
