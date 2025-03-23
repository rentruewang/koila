# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import functools
from abc import ABC
from typing import ClassVar

from aioway.errors import AiowayError

from .indices import Index
from .ops import IndexOp

__all__ = ["IndexTemplate"]


@dcls.dataclass(frozen=True, eq=False)
class IndexTemplate(ABC):
    """
    ``IndexTemplate`` corresponds to different types of backends, e.g. ``faiss``, ``b-tree``,
    and is responsible for routing to different ``Index`` types.
    """

    supported: ClassVar[tuple[type[IndexOp], ...]]
    """
    Yield a list of supported operators.
    These operators would then be called upon.

    Note:
        The type of the ``IndexOp``s are returned
        because ``IndexOp`` can be a family of operators.

        For example, roughly equal with different ``k``s.
    """

    @functools.cache
    def __call__(self, operator: IndexOp) -> type[Index]:
        if not isinstance(operator, self.supported):
            raise IndexTemplateNotSupportedError(
                f"{operator=} is not supported. ".capitalize()
                + f"Must be instances of {self.supported}"
            )

        return self[operator]

    @abc.abstractmethod
    def __getitem__(self, operator: IndexOp) -> type[Index]: ...


class IndexTemplateNotSupportedError(AiowayError, NotImplementedError): ...
