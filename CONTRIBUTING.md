# Contributing

## Development installation

First, clone and navigate into the project:

```bash
git clone https://github.com/rentruewang/aioway
cd aioway/
```

Alternatively, use ssh:
```bash
git clone git@github.com:rentruewang/aioway
cd aioway/
```

I'm using [pdm](https://pdm-project.org/) in this project for dependency management.
To install all dependencies (including development dependencies) with `pdm`, run

```bash
pdm install -G:all
```

Alternatively, use of `pip` is also allowed (although might be less robust due to lack of version solving)

```bash
pip install -e .
```

Both commands result in an editable installation.

## Recommended development style

### Python code style

The code style in the project closely follows the recommended standard of python:

1. [PEP8](https://peps.python.org/pep-0008/)
2. Class imports are non-qualified (`from module.path import ClassName`), and do not use unqualified function names (however, upper case functions acting as classes are treated as classes, lower case classes are treated as functions).
3. All other imports are qualified.
4. Use of absolute imports are preferred over relative imports except in the case where the other module is in the same package (`from .other_module import Something`).

### Documentation

The documentation string style follows the Google style format specified [here](https://mkdocstrings.github.io/griffe/docstrings/#google-style).

### Formatting

Use `autoflake`, `isort`, `black` for consistent formatting.

Prior to commiting, please run the following commands:

```bash
autoflake .
isort .
black .
```

### Commit message

Commit message should follow the format:

```
{emoji} Message. [([fix] #{issue})]

Detailed explanation.
```

Where `[]` denotes optional in the above commit message. `{emoji}` should be a relevant emoji, and `#{issue}` should be the relevant issue number.

### Typing

Be sure to run `mypy` prior to submitting! There can be issue with `mypy` not finding libraries. The command I use for checking is

```bash
mypy .
```
