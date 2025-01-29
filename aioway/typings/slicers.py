# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.errors import AiowayError

__all__ = ["Slicer"]


@dcls.dataclass(frozen=True)
class Slicer:
    """
    Processor for slice.
    """

    length: int
    """
    The length of the original sequence.
    """

    def __call__(self, sl: slice, /) -> slice:
        start = self._start_stop(sl.start, "start")
        stop = self._start_stop(sl.stop, "stop")
        step = sl.step or 1
        return slice(start, stop, step)

    def _start_stop(self, bnd: int, name: str) -> int:
        if bnd is None:
            return 0
        elif -self.length <= bnd < self.length:
            return bnd % self.length
        else:
            raise SliceError(f"Invalid slice {name} index: {bnd}")


class SliceError(AiowayError, ValueError): ...
