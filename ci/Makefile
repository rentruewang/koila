all: install build testing pre-commit

publish:
	python3 pdm.py publish

build:
	python3 pdm.py build

install:
	python3 pdm.py install

sync:
	python3 pdm.py sync

pre-commit:
	python3 pre-commit.py all

format:
	python3 pre-commit.py format

typing:
	python3 pre-commit.py typing

autoflake:
	python3 pre-commit.py autoflake

black:
	python3 pre-commit.py black

isort:
	python3 pre-commit.py isort

mypy:
	python3 pre-commit.py mypy

testing:
	python3 testing.py
