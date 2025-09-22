# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from pandas import DataFrame
from tensordict._td import TensorDict

from aioway._errors import AiowayError

from ._batches import BatchFrame

__all__ = ["PandasFrame"]


@dcls.dataclass(frozen=True)
class PandasFrame(BatchFrame, key="PANDAS"):
    """
    A ``pandas``-based ``Frame``,
    dynamically converting ``DataFrame`` to ``TensorDict``.

    Todo:
        This class is essentially a duplicate of the ``BatchFrame`` class.
        See how to merge the logics / data together.
    """

    KLASS = DataFrame

    @classmethod
    @typing.override
    def convert_tensordict(cls, data: DataFrame) -> TensorDict:
        return dataframe_to_tensordict(data)


def dataframe_to_tensordict(df: DataFrame, /, device: str = "cpu") -> TensorDict:
    """
    Converts ``DataFrame`` to ``TensorDict``.

    Note:
        For now, it's as simple as converting ``df`` to a ``list[dict]``,
        and reconstructing it with ``TensorDict``,
        without doing too much metadata manipulation.
        This can and should change in the future.
    """

    return TensorDict(df.to_dict("list"), device=device)


class PandasDataFrameTypeError(AiowayError, TypeError): ...
