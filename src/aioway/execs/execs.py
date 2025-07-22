# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from abc import ABC

import structlog

from aioway import registries
from aioway.errors import AiowayError
from aioway.nodes import Node

__all__ = ["Exec"]

LOGGER = structlog.get_logger()


@dcls.dataclass
class Exec(Node["Exec"], ABC):
    """
    ``Exec`` represents an executor.

    Upon launch (via ``iter``), it would kick off a stream of heterogenious data.

    It can be thought of as an ``Iterator`` of ``Block``s,
    where computation happens lazily, imperatively, and the result is yielded.
    """

    # poller: Poller
    # """
    # The strategy to poll data from the previous executors.
    # """

    def __init_subclass__(cls, *, key: str = ""):
        # Ensure that concrete subclasses can be instantiated from a factory.
        cls.__register_factory_with_key(cls, key=key)

    def __hash__(self) -> int:
        """
        The unique identifier of each node, representing computation.

        This means that ``__hash__`` would be the same for shared computation.

        For now, it is the ``id`` of the object itself,
        and the object itself is the unique identifier of the computation.
        However, in the future, we might want to use a more sophisticated way to identify computation
        for distributed execution and caching.
        """

        return id(self)

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    # @property
    # def children(self):
    #     return self.poller.children

    __register_factory_with_key = registries.init_subclass(lambda: Exec)
    """
    Register subclasses with the given key into the factory.
    """


class ExecRegisterError(AiowayError, KeyError): ...
