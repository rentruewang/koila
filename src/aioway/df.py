# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.schemas import ColumnSchema, TableSchema


@dcls.dataclass(frozen=True)
class Series:
    schema: ColumnSchema


class DataFrame:
    schema: TableSchema
