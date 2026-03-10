# Copyright (c) AIoWay Authors - All Rights Reserved

"The registry types using signatures as keys."

import logging
import typing
from collections.abc import Callable, Iterator, KeysView, Mapping
from typing import Self

from rich.table import Table

from .ops import Op, OpSign

__all__ = ["PerSignReg", "SignatureRegistry", "default_registry"]

LOGGER = logging.getLogger(__name__)


class PerSignReg(Mapping[str, Op]):
    """
    Per signature registry.

    This is a mapping from ``str -> Op``, by wrapping the registered ``Callable`` into ``Op``.
    """

    def __init__(self, signature: OpSign) -> None:
        """
        Args:
            signature: The signature that the registry must have.
        """

        self._signature = signature
        self._registry: dict[str, Callable] = {}

        LOGGER.info("Registry for %s created.", signature)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, key: object) -> bool:
        return key in self.keys()

    def __len__(self):
        return len(self._registry)

    @typing.override
    def __getitem__(self, key: str) -> Op:
        return Op(name=key, signature=self._signature, function=self._registry[key])

    def store(self, key: str, func: Callable) -> None:
        if key in self:
            raise KeyError(f"Cannnot set again for registry on {key=} with {func=}.")

        if not callable(func):
            raise ValueError(f"{func} is not callable.")

        # Unwrap in case there are nested ``Op``s.
        fn = func.__func__ if isinstance(func, Op) else func

        self._registry[key] = fn

    def keys(self) -> KeysView[str]:
        return self._registry.keys()


class SignatureRegistry(Mapping[OpSign, PerSignReg]):
    """
    The global registry that is based on signatures.

    This behaves like a default dict.
    """

    def __init__(self) -> None:
        self._regs: dict[OpSign, PerSignReg] = {}

        LOGGER.info("Global registry initialied.")

    def __contains__(self, key: object) -> bool:
        return key in self.keys()

    def __getitem__(self, signature: OpSign) -> PerSignReg:
        # DefaultDict behavior.
        if signature not in self:
            self._regs[signature] = PerSignReg(signature=signature)

        return self._regs[signature]

    def __len__(self) -> int:
        return len(self._regs)

    def __iter__(self) -> Iterator[OpSign]:
        return iter(self.keys())

    def __rich__(self):
        return _reg_rich_table(self)

    def keys(self) -> KeysView[OpSign]:
        return self._regs.keys()

    @property
    def signatures(self):
        return set(self.keys())

    @property
    def ops(self):
        return _unique_ops(self)

    def select(self, *signatures: OpSign) -> Self:
        "Only view the types of selected signatures."

        result = type(self)()

        for sign in signatures:
            result._regs[sign] = self[sign]

        return result


_REGISTRY = SignatureRegistry()


def default_registry():
    return _REGISTRY


def register(signature: OpSign, /, *keys: str):
    """
    Module private method for registering the callable based on their signature.

    The functionality is publically exposed in ``Signature.register_keys``.
    """

    registry = _REGISTRY[signature]

    def registrar[T: Callable](function: T) -> T:
        for key in keys:
            registry.store(key, function)

        return function

    return registrar


def dispatch(signature: OpSign, key: str, /) -> Op:
    """
    Module private method for dispaptching with signature and key.

    The functionality is publically exposed in ``Signature.dispatch``.
    """

    registry = _REGISTRY[signature]
    return registry[key]


def _reg_rich_table(registry: SignatureRegistry, /) -> Table:
    "A rich table of operator vs signature."

    signatures = list(registry.keys())
    table = Table(" ", *map(str, signatures))
    for op in sorted(_unique_ops(registry)):
        items = [registry[sign].get(op) for sign in signatures]
        table.add_row(op, *(i.name if i else "-" for i in items))
    return table


def _unique_ops(registry: SignatureRegistry, /) -> set[str]:
    result: set[str] = set()
    for per_type in registry.values():
        for op in per_type:
            result.add(op)
    return result
