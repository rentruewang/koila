# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.compilers import PreviewNode
from aioway.plans import Rewriter

from .tables import Table
from .volatile import Block


@dcls.dataclass(frozen=True)
class Runner(Rewriter[PreviewNode, Table]):
    source_blocks: dict[str, Block]

    def __call__(self, node: PreviewNode) -> Table:
        raise NotImplementedError
