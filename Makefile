.PHONY: development container
SHELL := /bin/bash

all: development

.ONESHELL:
development:
	@set -e; \
	if ! [ -x $(command -v poetry) ]; then \
		tput setaf 9; echo "Poetry is not installed, installing now..."; tput sgr0; \
		curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python; \
	fi

# Installing dependencies
	@tput setaf 3; echo "~ Running poetry install"; tput sgr0
	poetry install
	@tput setaf 3; echo "~ Installing pre-commit hooks"; tput sgr0
	poetry run pre-commit install

# Activate poetry environment
	poetry shell

# Set up a kernel
	poetry run python -m ipykernel install --user --name hangman_guesser

# Exit
	@tput setaf 2; tput bold; echo "~ Development environment successfully activated."; tput sgr0;


.ONESHELL:
submodule:
	@set -e
# Update submodule, fetch latest changes
	git submodule init
	git submodule update
