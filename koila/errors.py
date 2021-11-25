from typing import NoReturn


class NeverImplementedError(Exception):
    "Sorry, but this function won't get implemented, because it's dangerous or misleading."

    @classmethod
    def raise_error(cls, *args, **kwargs) -> NoReturn:
        del args
        del kwargs
        raise cls


class UnsupportedError(Exception):
    "Sorry, this function is currently not supported."
