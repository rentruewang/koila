# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections.abc import Iterator

import tensordict as td

from aioway import _common, fake
from aioway.fn import Fn

from .fn import TensorDictFn

__all__ = ["TensorDictDataFn"]


@_common.dcls_no_eq
class TensorDictDataFn(TensorDictFn):
    "The `Fn` representing a plain `td.TensorDict`."

    data: td.TensorDict

    def __post_init__(self) -> None:
        super().__init__()

        # Mark as `EVALUATED`.
        _ = self.do()

    @typing.override
    def forward(self) -> td.TensorDict:
        if not (mode := fake.is_enabled()):
            return self.data

        converter = mode.fake_tensor_converter
        return td.TensorDict(
            {
                key: converter.from_real_tensor(mode, value)
                for key, value in self.data.items()
            }
        )

    @typing.override
    def _deps(self) -> Iterator[Fn[td.TensorDict]]:
        return
        yield
