# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import logging
from importlib import metadata

from typer import Typer

from aioway import common

app = Typer(pretty_exceptions_enable=not common.compiled())


@app.command()
def help() -> None:
    VERSION = metadata.version("aioway")
    LOGO = r"""
     The Goal Oriented Machine Learning Platform


     █████  ██  ██████  ██     ██  █████  ██    ██
    ██   ██ ██ ██    ██ ██     ██ ██   ██  ██  ██
    ███████ ██ ██    ██ ██  █  ██ ███████   ████
    ██   ██ ██ ██    ██ ██ ███ ██ ██   ██    ██
    ██   ██ ██  ██████   ███ ███  ██   ██    ██


                Version: {version}
            Compiled (yes/no): {compiled}
""".format(
        version=VERSION, compiled="yes" if common.compiled() else "no"
    )

    print(LOGO)


if __name__ == "__main__":
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.INFO)
    app()
