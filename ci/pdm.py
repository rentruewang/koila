# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import subprocess as sp
from argparse import ArgumentParser
from collections.abc import Callable

import gha
import sh


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

    with sh.run_in_root():
        sh.cmd(f"pdm {command}")


@ensure("pdm")
def sync() -> None:
    "Sync the dependencies. Must have ``pdm.lock`` present."

    cmd = "sync -G:all"

    if gha.in_github_actions():
        cmd = f"{cmd} -v"

    command(cmd)


@ensure("pdm")
def install() -> None:
    "Install the dependencies. Might update ``pdm.lock``."

    cmd = "install -G:all"

    if gha.in_github_actions():
        cmd = f"{cmd} -v"

    command(cmd)


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


def parse_args():

    parser = ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("args", nargs="*")
    flags = vars(parser.parse_args())

    @dcls.dataclass(frozen=True)
    class ParsedArgs:
        cmd: str
        args: list[str] = dcls.field(default_factory=list)

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
