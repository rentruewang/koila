# Copyright (c) AIoWay Authors - All Rights Reserved

from .errors import AiowayError

__all__ = ["PlannedButNotImplemented"]


class PlannedButNotImplemented(AiowayError, NotImplementedError):
    """
    This error is raised when a feature is planned, but not implemented.
    """
