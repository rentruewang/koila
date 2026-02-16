# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import subprocess as sp
import sys
from argparse import ArgumentParser
from collections.abc import Callable
from typing import Literal

import gha
import sh


def in_venv() -> bool:
    "Check if we are running inside a venv."

    return sys.prefix != sys.base_prefix


def ensure(binary: str, /):
    "Ensure that the binary is installed for the wrapped function."

    def wrapper[**P, T](wrapped: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(wrapped)
        def func(*args: P.args, **kwargs: P.kwargs) -> T:
            _check_binary(binary)
            return wrapped(*args, **kwargs)

        return func

    return wrapper


@functools.cache
def _check_binary(binary: str, /) -> None:
    print(f"Trying to check if '{binary}' is installed...", end=" ")

    try:
        _ = sp.run([binary], check=True, capture_output=True)
        print("Yes")
    except:
        print("No")
        raise


@ensure("pdm")
def command(command: str, /) -> None:
    "Call a ``pdm`` command."

    with sh.run_in_project_root():
        sh.cmd(f"pdm {command}")


@ensure("pdm")
def _setup_deps(mode: Literal["sync", "install"], /, extras: bool = True) -> None:
    # No need to re-setup.
    if in_venv():
        return None

    cmd: list[str] = [mode]

    if extras:
        cmd.append("-G:all")

    if gha.in_github_actions():
        cmd.append("-v")

    command(" ".join(cmd))


def sync() -> None:
    "Sync the dependencies. Must have ``pdm.lock`` present."

    _setup_deps("sync")


def install() -> None:
    "Install the dependencies. Might update ``pdm.lock``."

    _setup_deps("install")


@ensure("pdm")
def run(cmd: str, /) -> None:
    "Run the sub command using ``pdm``'s environment."

    command(f"run {cmd}")


@ensure("pdm")
def build() -> None:
    "Build the wheel into the ``dist`` folder."

    command("build")


@ensure("pdm")
def publish() -> None:
    "Publish the built wheel into PyPI."

    command("publish")


@dcls.dataclass(frozen=True, kw_only=True)
class ParsedArgs:
    cmd: str
    args: list[str] = dcls.field(default_factory=list)


def parse_args():

    parser = ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("args", nargs="*")
    flags = vars(parser.parse_args())

    return ParsedArgs(cmd=flags["command"], args=flags["args"])


if __name__ == "__main__":
    flags = parse_args()

    # Run would run an arbitrary command, supplied by ``args``.
    if flags.cmd == "run":
        run(" ".join(flags.args))

    # The rest are all pre-defined workflows.
    else:
        assert not flags.args
        globals()[flags.cmd]()
