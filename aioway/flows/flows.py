# Copyright (c) RenChu Wang - All Rights Reserved

from aioway.execs import Exec


class Flow:
    """
    Base class for all flows.

    A flow is a sequence of operations that can be executed in parallel or in sequence.
    """

    def __call__(self, executor: Exec) -> None:
        for _ in executor:
            pass
