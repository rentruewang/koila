all: install build testing pre-commit

publish:
	python3 pdm.py publish

build:
	python3 pdm.py build

install:
	python3 pdm.py install

sync:
	python3 pdm.py sync

pre-commit: format typing

format:
	python3 pre-commit.py format

typing:
	python3 pre-commit.py typing

# Linting is included, but not on by default.
linting:
	python3 pre-commit.py linting

testing:
	python3 testing.py
