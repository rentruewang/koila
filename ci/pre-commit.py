# Copyright (c) AIoWay Authors - All Rights Reserved

from argparse import ArgumentParser

import gha
import pdm
import sh


def parse_args() -> str:

    parser = ArgumentParser()
    parser.add_argument("command", type=str)

    flags = vars(parser.parse_args())

    return flags["command"]


def commands_to_run(command: str, /):
    match command:
        case "all":
            yield from commands_to_run("typing")
            yield from commands_to_run("format")
        case "autoflake" | "isort" | "black":
            yield f"{command} ."
        case "mypy" | "typing":
            yield "mypy src/"
        case "format":
            yield "autoflake ."
            yield "isort ."
            yield "black ."
        case _:
            raise NotImplementedError(
                f"Support for '{command}' command is not yet implemented."
            )


def run_tools(command: str) -> list[str]:
    failures: list[str] = []
    for tool in commands_to_run(command):
        try:
            pdm.run(tool)
        except Exception:
            failures.append(tool)
    return failures


if __name__ == "__main__":
    gha.setup()
    pdm.sync()
    command = parse_args()

    with sh.run_in_project_root():
        failed_tools = run_tools(command)

    if failed_tools:
        failed = ",".join(map(lambda tool: f"'{tool}'", failed_tools))
        raise RuntimeError(f"Tools that failed execution: {failed}")
