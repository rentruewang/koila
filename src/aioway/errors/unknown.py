# Copyright (c) AIoWay Authors - All Rights Reserved

from .errors import AiowayError


class UnknownTypeError(AiowayError, TypeError): ...
