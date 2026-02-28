# Copyright (c) AIoWay Authors - All Rights Reserved

import functools
import os

import sh


def remove_unwanted_files() -> None:
    "Remove the files GitHub Actions pre-installed."

    print("Removing files we did not ask for...")

    for folder in [
        "/usr/local/lib/android",
        "/usr/share/dotnet",
        "/usr/local/.ghcup",
    ]:
        sh.cmd(f"sudo rm -rf {folder}")

    sh.cmd("docker system prune -af --volumes")


def log_storage_usage() -> None:
    "Log how much usage is currently being used by GitHub Actions."
    print("Investigating how much storage is used in GitHub Actions...")

    sh.cmd("df -h")


@functools.cache
def in_github_actions() -> bool:
    "Detect whether or not it is running in GitHub Actions."

    print("Checking if we are in GitHub Actions...", end=" ")
    result = os.getenv("GITHUB_ACTIONS") == "true"
    print("Yes" if result else "No")
    return result


@functools.cache
def setup() -> None:
    "The shared entrypoint to GitHub Actions scripts"

    # Does nothing outside of GitHub Actions.
    if not in_github_actions():
        return

    remove_unwanted_files()
    log_storage_usage()
