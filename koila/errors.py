from typing import NoReturn


class UnsupportedError(RuntimeError):
    "Sorry, this function is currently not supported."

    @classmethod
    def raise_error(cls, *args, **kwargs) -> NoReturn:
        del (args, kwargs)
        raise cls
