from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol


class Indexible(Protocol):
    @abstractmethod
    def __getitem__(self, index: Any) -> Any:
        ...

    @abstractmethod
    def __setitem__(self, index: Any, value: Any) -> None:
        ...
