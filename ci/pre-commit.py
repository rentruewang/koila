# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from argparse import ArgumentParser
from collections.abc import Generator

import gha
import pdm
import sh


@dcls.dataclass(kw_only=True)
class Args:
    command: str
    dry_run: bool


def parse_args() -> Args:

    parser = ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("-n", "--dry-run", action="store_true")

    flags = parser.parse_args()

    return Args(command=flags.command, dry_run=flags.dry_run)


@dcls.dataclass(kw_only=True)
class Tool:
    "The tools and their descriptions."

    name: str
    "The name of the tool."

    cmd: str
    "The command that actually runs."


def commands_to_run(command: str, /) -> Generator[Tool]:
    match command:
        case "all":
            yield from commands_to_run("format")
            yield from commands_to_run("typing")
        case "autoflake" | "isort" | "black":
            yield Tool(name=command, cmd=f"{command} .")
        case "mypy" | "typing":
            yield Tool(name="mypy", cmd="mypy src/")
        case "format":
            for cmd in ["autoflake", "isort", "black"]:
                yield from commands_to_run(cmd)
        case _:
            raise NotImplementedError(
                f"Support for '{command}' command is not yet implemented."
            )


def run_tools(args: Args) -> list[str]:
    "Execute the tools, return the tools that failed."

    # Maybe run the command. Depends on the ``dry_run`` flag.
    run = print if args.dry_run else pdm.run

    failures: list[str] = []
    for tool in commands_to_run(args.command):
        try:
            run(tool.cmd)
        except Exception:
            failures.append(tool.name)
    return failures


if __name__ == "__main__":
    gha.setup()
    pdm.install()

    args = parse_args()

    with sh.run_in_project_root():
        failed_tools = run_tools(args)

    if failed_tools:
        failed = ",".join(map(lambda tool: f"'{tool}'", failed_tools))
        raise RuntimeError(f"Tools that failed execution: {failed}")
