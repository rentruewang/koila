from typing import NoReturn


class UnsupportedError(Exception):
    "Sorry, this function is currently not supported."

    @classmethod
    def raise_error(cls, *args, **kwargs) -> NoReturn:
        del args
        del kwargs
        raise cls
