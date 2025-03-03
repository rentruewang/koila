# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections.abc import Iterator

from tensordict import TensorDict

from aioway.frames import Frame
from aioway.streams import Stream


class Exec(Iterator[TensorDict], ABC):
    def __next__(self) -> TensorDict: ...


@dcls.dataclass(frozen=True)
class LeafExec(Exec, ABC): ...


@dcls.dataclass(frozen=True)
class UnaryExec(Exec, ABC):
    child: Exec


@dcls.dataclass(frozen=True)
class BinaryExec(Exec, ABC):
    left: Exec
    right: Exec


# OK, so we do not need or want Exec to depend on Frame or Stream,
# simply based on the fact that Exec is higher level than Frame or Stream.
# This is to say, Exec can produce Frame or Stream,
# but not depend on them.
# TODO: make changes to this.

# @dcls.dataclass(frozen=True)
# class UnaryFrameExec(Exec, ABC):
#     child: Frame


# @dcls.dataclass(frozen=True)
# class UnaryStreamExec(Exec, ABC):
#     child: Stream


# @dcls.dataclass(frozen=True)
# class BinaryFrameExec(Exec, ABC):
#     left: Frame
#     right: Frame


# @dcls.dataclass(frozen=True)
# class BinaryStreamExec(Exec, ABC):
#     left: Stream
#     right: Stream
