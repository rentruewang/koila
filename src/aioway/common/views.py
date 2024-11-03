# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Callable, Iterator, Mapping
from typing import Generic, TypeVar

from .typing import KeysAndGetItem

_K = TypeVar("_K")
_V = TypeVar("_V")
_T = TypeVar("_T")


class MapTransform(Mapping[_K, _T], Generic[_K, _V, _T]):
    """
    A view class that allows to map a function over the values of a mapping.
    """

    def __init__(self, mapping: Mapping[_K, _V], proc: Callable[[_V], _T]) -> None:
        """
        Args
            mapping: The mapping object to be viewed.
            proc: The function to be applied to the values of the mapping.
        """

        assert isinstance(mapping, KeysAndGetItem)

        self._mapping = mapping
        self._proc = proc

    def __len__(self) -> int:
        return len(self._mapping)

    def __getitem__(self, key: _K) -> _T:
        item = self._mapping[key]
        return self._proc(item)

    def __iter__(self) -> Iterator[_K]:
        return iter(self._mapping)

    def __contains__(self, key: object) -> bool:
        return key in self._mapping
