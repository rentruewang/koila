# Copyright (c) RenChu Wang - All Rights Reserved

from .errors import AiowayError


class UnknownTypeError(AiowayError, TypeError): ...
