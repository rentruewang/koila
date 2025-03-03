# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Iterator

from tensordict import TensorDict

from aioway.frames import Frame

from .samplers import Sampler
from .streams import Stream

__all__ = ["FrameStream"]


@dcls.dataclass(frozen=True)
class FrameStream(Stream):
    frame: Frame
    """
    The underlying dataframe that contains the data.
    """

    sampler: Sampler
    """
    The sampler for which to access the original frame.
    """

    def __iter__(self) -> Iterator[TensorDict]:
        for indices in self.sampler:
            yield self.frame[indices]
