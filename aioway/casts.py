# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC
from typing import LiteralString


class Castable[C: "Castable"](ABC):
    @abc.abstractmethod
    def cast(self, target: LiteralString) -> C: ...
