# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections.abc import Iterable, Sequence
from typing import Generic, Literal, TypeVar

_T = TypeVar("_T")
_N = TypeVar("_N", int, float)


@dcls.dataclass(frozen=True)
class Range(Generic[_T], ABC):
    """
    A closed interval class that checks if a value falls within the given interval or not.
    """

    default: _T | None = None
    """
    The default value (if given) or ``None`` if not given.
    """

    def __post_init__(self):
        if (default := self.default) is not None and default not in self:
            raise ValueError(
                f"The default value specified {self.default=} not in self."
            )

    @abc.abstractmethod
    def __contains__(self, value: _T) -> bool:
        """
        Check if the given value falls within this interval.
        """


@dcls.dataclass(frozen=True)
class Interval(Range[_N], ABC):
    _: dcls.KW_ONLY

    lower: _N
    """
    The lower bound of the tunable range.
    """

    upper: _N
    """
    The upper bound of the tuable range.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.lower > self.upper:
            raise ValueError(f"{self.lower=} is not smaller or equal to {self.upper=}.")

    def __contains__(self, value: _N) -> bool:
        return self.lower <= value <= self.upper

    def __lt__(self, value: _N) -> bool:
        return value < self.lower

    def __le__(self, value: _N) -> bool:
        return value <= self.upper

    def __gt__(self, value: _N) -> bool:
        return value > self.upper

    def __ge__(self, value: _N) -> bool:
        return value >= self.lower


@dcls.dataclass(frozen=True)
class Choice(Range[_T], ABC):
    _: dcls.KW_ONLY

    options: Sequence[_T]

    def __len__(self) -> int:
        return len(self.options)

    def __contains__(self, value: _T) -> bool:
        return value in self.options


@dcls.dataclass(frozen=True)
class IntInterval(Interval[int]):
    def __len__(self) -> int:
        return self.upper - self.lower + 1

    def __iter__(self) -> Iterable[int]:
        return iter(range(self.lower, self.upper + 1))


@dcls.dataclass(frozen=True)
class FloatInterval(Interval[float]):
    pass


@dcls.dataclass(frozen=True)
class BoolChoice(Choice[bool]):
    options: tuple[Literal[False], Literal[True]] = False, True


@dcls.dataclass(frozen=True)
class StrChoice(Choice[str]):
    options: list[str] = dcls.field(default_factory=list)
