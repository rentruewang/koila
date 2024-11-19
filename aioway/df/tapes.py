# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from typing import Protocol


@typing.runtime_checkable
class HasTape(Protocol):
    _tape: "Tape"


@dcls.dataclass(frozen=True)
class Tape: ...
