# Copyright (c) RenChu Wang - All Rights Reserved

from typing import Literal

from pandas import DataFrame

from .blocks import Block


def block(
    df: DataFrame, kind: Literal["dataframe", "tensordict"] = "tensordict"
) -> Block:
    raise NotImplementedError
