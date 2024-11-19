# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc

from aioway.tables import Table


class CompileToTable:
    @abc.abstractmethod
    def compile(self) -> Table: ...
