# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import inspect

from aioway import common


def func_with_doc():
    """
    Documentation of this function used for testing the `common.wraps` function.
    """


@common.wraps(func_with_doc)
def wrapped_func():
    return func_with_doc()


def test_function_wrapper():
    assert func_with_doc.__doc__ == wrapped_func.__doc__
    assert inspect.signature(func_with_doc) == inspect.signature(wrapped_func)
