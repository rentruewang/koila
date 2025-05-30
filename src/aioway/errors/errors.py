# Copyright (c) AIoWay Authors - All Rights Reserved

import hashlib
from typing import Any

__all__ = ["AiowayError"]


class AiowayError(Exception):
    """
    ``AiowayError`` is the error thrown by the ``aioway`` library.
    """

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)

    def __str__(self):
        super_str = super().__str__()
        hash_code = self.__hash_code()

        message = f"[{hash_code}]"

        if super_str:
            message = f"{message} {super_str}"

        return message

    @classmethod
    def __hash_code(cls) -> str:
        if result := _MD5_CACHE.get(cls, None):
            return result

        unique = cls._unique_string()
        md5 = _hash_code(unique)
        _MD5_CACHE[cls] = md5
        return md5

    @classmethod
    def _unique_string(cls) -> str:
        """
        Create a unique string (yet reproducible accross sessions and platforms)
        for an ``AiowayError`` type. This is useful in identifying which error is where.

        Returns:
            The string that wuold be hashed with ``md5`` algorithm.
        """

        if not issubclass(cls, AiowayError):
            raise TypeError(
                "Class must be a subclass of `AiowayError` to use this utility."
            )

        bases = cls.__bases__
        string = "\n".join([cls.__name__, *(base.__name__ for base in bases)])
        return string


_MD5_CACHE: dict[type[AiowayError], str] = {}


def _hash_code(string: str) -> str:
    hashed = hashlib.md5(string.encode("utf-8"))
    hash_code = hashed.hexdigest()
    return f"{hash_code[0:4]}-{hash_code[4:8]}"
