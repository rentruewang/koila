# Copyright (c) AIoWay Authors - All Rights Reserved

from typing import ClassVar, Protocol


class Nargs(Protocol):
    N_ARY: ClassVar[int]
