# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from types import NoneType

import pytest

from aioway.previews import Info, Preview, Registry
from aioway.relalg import Relation
from aioway.schemas import DataTypeEnum, Einsum


@pytest.fixture(scope="function")
def registry() -> Registry[None]:
    return Registry[None]()


@dcls.dataclass(frozen=True)
class EinsumPreview(Preview):
    einsum: Einsum

    def _compute(self, *infos: Info) -> Info:
        if len(infos) != len(self.einsum.inputs):
            return NotImplemented

        in_shapes = [info.shape for info in infos]
        [out_shape] = self.einsum(in_shapes)
        return Info(out_shape, dtype=DataTypeEnum.FLOAT())


@pytest.fixture
def preview(relation: type[Relation], num_params: int, einsum: str):
    return EinsumPreview(
        relation=relation, num_params=num_params, einsum=einsum, initialization=NoneType
    )


@pytest.mark.parametrize("einsums", [("ab->a",), ("ab->a", "ab->b", "ab,cd->abcd")])
def test_registry(registry, einsums):
    for einsum in einsums:
        registry.register(
            EinsumPreview(
                einsum=einsum, relation=Relation, num_params=0, initialization=NoneType
            )
        )

    for preview in registry:
        assert preview.initialization() is None
