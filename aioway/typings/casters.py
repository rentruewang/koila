# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import functools
from abc import ABC
from collections.abc import Callable, Sequence
from typing import LiteralString, Self

import numpy as np

from aioway.errors import AiowayError

__all__ = ["Caster", "Castable"]


def _checked[**P, T, E: Exception](unchecked: Callable[P, T], *err_types: type[E]):
    assert callable(unchecked)

    def checked(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return unchecked(*args, **kwargs)
        except err_types as e:
            raise CasterInternalError("Internal error encountered.") from e

    checked.__doc__ = unchecked.__doc__
    return checked


@dcls.dataclass(frozen=True)
class Caster:
    """
    The casting manager for a class.

    Fixme:
        Work out the type issues documented in python/mypy#4717,
        so that type hints can work properly.
    """

    base: type
    """
    The base class of the caster.
    """

    aliases: Sequence[str]
    """
    The aliases of the underlying type used in ``cast``.
    """

    klasses: Sequence[type]
    """
    The sub classes of the caster.
    """

    matrix: Sequence[Sequence[Callable | None]]
    """
    The matrix contains the instructions for casting from a class to another class.
    """

    def __post_init__(self) -> None:
        for sub in self.klasses:
            if sub == self.base:
                raise CasterInitError(
                    f"Base class: {self.base} must be abstract, "
                    f"and cannot be defined in the subclass list: {self.klasses}"
                )

            if issubclass(sub, self.base):
                continue

            raise CasterInitError(
                "Casting only works when "
                f"{sub} is a subclass of (<:) {self.base}. "
                "This is not the case."
            )

        if len(set(self.aliases)) != len(self.aliases):
            raise CasterInitError("Aliases must be unique!")

        if not all(isinstance(name, str) for name in self.aliases):
            raise CasterInitError("Aliases must be strings!")

        if len(self.aliases) != len(self.klasses):
            raise CasterInitError("Aliases must be of the same length as the types.")

        try:
            array = np.array(self.matrix)
        except ValueError as ve:
            raise CasterInitError(
                "Matrix given must be convertable to a numpy array."
            ) from ve

        if array.shape[0] != array.shape[1]:
            raise CasterInitError("Matrix given must be sqaure")

    def __getitem__(self, idx: tuple[str, str]) -> Callable | None:
        """
        A convenience wrapper for matrix, allowing getting from the matrix.
        Returns the conversion function from the matrix.
        """

        try:
            source, target = idx
        except ValueError as ve:
            raise CasterInitError(
                f"Cannot unpack {idx} as a tuple of source and target."
            ) from ve

        src_idx = self.__alias_order(source)
        tgt_idx = self.__alias_order(target)

        try:
            return self.matrix[src_idx][tgt_idx]
        except KeyError as ke:
            raise CasterInitError(f"Mapping {source} -> {target} not found.") from ke

    def __name_order_unchecked(self, alias: str) -> int:
        return self.aliases.index(alias)

    def __klass_order_unchecked(self, klass: type) -> int:
        return self.klasses.index(klass)

    def __alias_of_unchecked(self, klass: type) -> str:
        index = self.__klass_order(klass)
        return self.aliases[index]

    def __lookup_unchecked(self, alias: str) -> type:
        index = self.__alias_order(alias)
        return self.klasses[index]

    __alias_order = _checked(__name_order_unchecked, ValueError)
    __klass_order = _checked(__klass_order_unchecked, ValueError)

    alias_of = _checked(__alias_of_unchecked, IndexError)
    lookup = _checked(__lookup_unchecked, IndexError)


class Castable(ABC):
    """
    A mixin that allows the use of ``cast`` like ``numpy``'s ``astype``.
    """

    @classmethod
    @abc.abstractmethod
    def _caster(cls) -> Caster:
        """
        The caster that supports translating between different subclasses.
        Must be implemented in the base class.
        """

        ...

    @classmethod
    @functools.cache
    def _type_cast(cls) -> Caster:
        return cls._caster()

    def cast(self, to: LiteralString, /) -> Self:
        """
        Attempt to convert to the given aliased type.

        Args:
            to:
                The given dtype to convert to.
                If ``to`` is the same as the alias of the current type,
                and that the self translation is not defined (``None`` in ``matrix``),
                no conversion is made and ``self`` is returned.

        Raises:
            TypeCastError:
                If there is not a suitable conversion found.

        Returns:
            The converted instance.
        """

        caster = self._type_cast()
        alias = caster.alias_of(type(self))

        # Do conversion if there is a direct translation.
        if f := caster[alias, to]:
            result = f(self)

            if not isinstance(result, caster.lookup(to)):
                raise TypeCastError(
                    f"Conversion function outputs a different type: {type(result)},"
                    f" rather than the target type: {caster.lookup(to)}."
                )

            return result

        # No conversion function, but no conversion needed,
        # because source type is the same as target type.
        if self._type_alias() == to:
            return self

        raise TypeCastError(f"Type casting from '{alias}' to '{to}' is not defined.")

    def _type_alias(self) -> bool:
        caster = self._type_cast()
        return caster.alias_of(type(self))


class CasterInitError(AiowayError, TypeError): ...


class CasterInternalError(AiowayError, RuntimeError): ...


class TypeCastError(AiowayError, TypeError): ...
