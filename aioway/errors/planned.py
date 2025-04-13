# Copyright (c) RenChu Wang - All Rights Reserved

__all__ = ["PlannedButNotImplemented"]

from .errors import AiowayError


class PlannedButNotImplemented(AiowayError, NotImplementedError):
    """
    This error is raised when a feature is planned, but not implemented.
    """
