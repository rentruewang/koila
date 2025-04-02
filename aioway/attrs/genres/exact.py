# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing

from aioway.attrs.sets import AttrSet

from .genres import AttrGenre


@dcls.dataclass(frozen=True)
class ExactAttrGenre(AttrGenre):
    attrs: AttrSet

    @typing.override
    def contains(self, attrs: AttrSet) -> bool:
        return attrs == self.attrs


@dcls.dataclass(frozen=True)
class SubsetAttrGenre(AttrGenre):
    attrs: AttrSet

    @typing.override
    def contains(self, attrs: AttrSet) -> bool:
        # Impossible to be subset this way.
        if len(attrs) > len(self.attrs):
            return False

        if not set(attrs.keys()).issubset(self.attrs.keys()):
            return False

        return all(attrs[key] == self.attrs[key] for key in attrs.keys())
