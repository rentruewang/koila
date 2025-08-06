# Copyright (c) AIoWay Authors - All Rights Reserved

"""
Casting between types
#####################

Different data has different data types,
which is central to how ``aioway`` works under the hood.

Often times data structures between different implementations don't work well together,
either cannot be used together (``numpy`` and ``torch``),
or doesn't have a compatible API (``pandas`` and ``numpy``).

This simply means that we would need to do our own casting and providing a unified business logic API.

So, how do we do it?

Assumption
**********

When we are using a framework,
we make the assumption that using operations a framework supports natively should be the fastest.

This is to say, having no external operation (like serializing and deserializing) is faster.

For example, using the different list interfaces as examples;
if we have a family of classes e.g. array list and linked list,
normally we can convert them by hand or just transfer data in between “normally”.

However, doing so means serializing and deserializing from device if the data is not locally on the same machine,
or if serializing and deserializing is expensive e.g. numpy or torch,
and this is where auto casting can help.
This way numpy / torch can auto cast to the desired dtype.

In this case, the frameworks are the classes (numpy array),
and non native operations like loading into python slows it down.

By doing auto casting, we optimize away the part where we would need to have to cast things manually.

Primitive types also has this.
And this should be completely hidden from the user.
Otherwise, user casting has more benefit.

Efficiency vs complexity
************************

If we define different subtypes of our phsyical abstraction, a neccessity to avoid translation costs,
we would need multiple implementations over the same class (for example, ``Block``).

However, this comes at the cost of complexity, even mentally,
because we need to define all the methods converting to different types on all classes.

However, if we draw inspiration from ``numpy``'s ``astype`` method,
which allows users to convert between different data types,
while only needing to provide 1 global matrix (type conversion matrix).

For this purpose, I have written a type called ``Caster``,
providing the convenient ``astype`` methods with a standardized interface.

Or maybe deal with the translation cost
***************************************

Internally, whenever we are translating from ``numpy`` to ``torch``, there is a cost.

If we are converting from ``ibis`` to ``torch``,
the cost is much larger because we would need to pull everything into memory, uncompressed.

However, during the prototype phase, the cost may be negligible,
compared to the multiple class changes we have to do everytime someone updates the API.

I think for now, let's simplify the costs first and then later add them back.
"""

import abc
import dataclasses as dcls
import functools
import logging
from abc import ABC
from collections.abc import Callable, Sequence
from typing import LiteralString, Self

import numpy as np

from aioway.errors import AiowayError

__all__ = ["Caster", "Castable"]

LOGGER = logging.getLogger(__name__)


def _checked[**P, T, E: Exception](unchecked: Callable[P, T], *err_types: type[E]):
    assert callable(unchecked)

    def checked(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            LOGGER.debug(
                "Passing computation to %s with args=%s and kwargs=%s",
                unchecked,
                args,
                kwargs,
            )

            return unchecked(*args, **kwargs)
        except err_types as e:
            raise CasterInternalError("Internal error encountered.") from e

    checked.__doc__ = unchecked.__doc__
    return checked


@dcls.dataclass(frozen=True)
class Caster:
    """
    The casting manager for a class.

    Note:
        Once type issues in python/mypy#4717 is fixed,
        don't use just ``type``, but a ``TypeVar``.

    Todo:
        Make using this class easier.
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
            # If the convertion type is the base class.
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

        LOGGER.debug("Matrix getitem called with idx=%s", idx)
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
