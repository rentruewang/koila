# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from typing import Protocol

from aioway.schemas import DataType


class Columnar(Protocol):
    @abc.abstractmethod
    def dtype(self) -> DataType: ...
