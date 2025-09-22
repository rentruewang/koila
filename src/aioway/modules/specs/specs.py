# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections.abc import Iterator, Sequence

from sympy import Interval

from aioway._errors import AiowayError

__all__ = ["Spec", "IntervalSpec", "ChoiceSpec"]

type Primitive = int | float | bool


class Spec(ABC):
    """
    Base class for all specs.

    A spec is a constraint on the parameter of an object,
    used to check if the parameter is valid.
    """

    @abc.abstractmethod
    def __contains__(self, obj: object) -> bool:
        """
        Check if the object is in the spec.
        """

        ...

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Return the string representation of the spec.
        """

        ...


@dcls.dataclass(frozen=True)
class IntervalSpec[T: Primitive](Spec):
    """
    A spec that checks if an object is in a range.

    """

    dtype: type[T]
    """
    The type of the object.
    """

    lower: T | None = None
    """
    The lower bound of the interval.
    """

    lower_eq: bool = False
    """
    If true, value can be equal to the lower bound.
    """

    upper: T | None = None
    """
    The upper bound of the interval.
    """

    upper_eq: bool = False
    """
    If true, value can be equal to the upper bound.
    """

    def __contains__(self, other: object, /) -> bool:
        return other in self.to_sympy()

    def __str__(self) -> str:
        # If the bounds aren't given, the bound is oo,
        # and we use closed interval for notation.
        lower_bound = "[" if self.lower is None or self.lower_eq else "("
        upper_bound = "]" if self.upper is None or self.upper_eq else ")"

        lower_val = self.lower if self.lower is not None else "-oo"
        upper_val = self.upper if self.upper is not None else "oo"

        return f"{lower_bound}{lower_val}, {upper_val}{upper_bound}"

    def to_sympy(self) -> Interval:
        return Interval(
            start=self.lower,
            end=self.upper,
            left_open=not self.lower_eq,
            right_open=not self.upper_eq,
        )


@dcls.dataclass(frozen=True)
class ChoiceSpec[T: Primitive](Spec):
    """
    A spec that checks if an object is in a list of choices.
    """

    options: Sequence[T]
    """
    The list of choices.
    """

    dtype: type[T]
    """
    The type of the object.
    This is used to check if the object is of the correct type.

    For now, the assumption is that all elements in ``options`` are of the same type.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.options, Sequence):
            raise ChoiceTypeError("options must be a sequence")

        if not all(isinstance(o, self.dtype) for o in self.options):
            raise ChoiceTypeError(
                f"All elements in options must be of type {self.dtype.__name__}"
            )

    def __iter__(self) -> Iterator[T]:
        yield from self.options

    def __contains__(self, object: object) -> bool:
        if not isinstance(object, self.dtype):
            return False

        return object in self.options

    def __str__(self) -> str:
        return str(list(self.options))


class ChoiceTypeError(AiowayError, TypeError): ...
