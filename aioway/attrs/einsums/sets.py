# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import logging
from collections.abc import Callable
from typing import Self

from aioway.attrs.attrs import AttrSet, NamedAttr
from aioway.errors import AiowayError

from .attrs import EinsumAttrFunc, EinsumDevice, EinsumDType, EinsumMap, EinsumShape
from .parsers import EinsumParser

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class EinsumAttrSetFunc:
    """
    ``EinsumAttrSetFunction`` processes an ``AttrSet`` into another ``AttrSet``.

    This is useful in predicting what an operator would behave before invoking it.
    """

    names: EinsumMap
    """
    The name mappings to follow.
    """

    shapes: EinsumShape
    """
    The shape ``Einsum``.
    """

    devices: EinsumDevice
    """
    The device ``Einsum``.
    """

    dtypes: EinsumDType
    """
    The dtype ``Einsum``.
    """

    def __post_init__(self) -> None:
        self.__check_input_or_output("input", lambda f: f.num_inputs)
        self.__check_input_or_output("output", lambda f: f.num_outputs)

    def __check_input_or_output(
        self, name: str, callback: Callable[[EinsumAttrFunc], int]
    ) -> None:
        LOGGER.debug("Checking %s stage with callback: %s", name, callback)

        assert callable(callback), "Callback given: {} is not callable.".format(
            callback
        )

        if len({callback(attr) for attr in self.attrs}) != 1:
            raise EinsumArgumentError(
                f"{name.capitalize()} of attributes do not share the same number of parameters."
            )

    def _compute(self, attrs: AttrSet) -> AttrSet:
        """
        Whether or not the ``AttrSet`` is computable.
        """

        names = self.names(tuple(attrs.names))
        shapes = self.shapes(tuple(attrs.shapes))
        devices = self.devices(tuple(attrs.devices))
        dtypes = self.dtypes(tuple(attrs.dtypes))

        return AttrSet.from_iterable(
            NamedAttr(name=name, shape=shape, device=device, dtype=dtype)
            for name, shape, device, dtype in zip(names, shapes, devices, dtypes)
        )

    @property
    def attrs(self) -> list[EinsumAttrFunc]:
        return [self.names, self.shapes, self.devices, self.dtypes]

    @classmethod
    def parse(
        cls,
        *,
        names: str,
        shapes: str,
        devices: str,
        dtypes: str,
        parser: EinsumParser = EinsumParser.init(),
    ) -> Self:
        return cls(
            names=parser(names),
            shapes=parser(shapes),
            devices=parser(devices),
            dtypes=parser(dtypes),
        )


class EinsumArgumentError(AiowayError, ValueError): ...
