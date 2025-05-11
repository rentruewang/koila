# Copyright (c) RenChu Wang - All Rights Reserved

from typing import Protocol


class Loggable(Protocol):
    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...
