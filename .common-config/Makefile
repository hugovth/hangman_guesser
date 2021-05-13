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

# Download source map
	@tput setaf 3; echo "~ Downloading source map"; tput sgr0
	@aws s3 cp s3://dash-data-team/common-files/sources_map.yaml properties/sources_map.yaml

# Activate poetry environment
	poetry shell

# Set up a kernel
	poetry run python -m ipykernel install --user --name development

# Exit
	@tput setaf 2; tput bold; echo "~ Development environment successfully activated."; tput sgr0;


.ONESHELL:
container:
	@set -e; \
	if [ -z "$(name)" ]; then \
		tput setaf 9; tput bold; echo "Please provide the docker image 'name' variable e.g. name=post-processing:latest"; tput sgr0; \
		exit 1; \
	fi

# Log into container registry
	tput setaf 3; echo "~ Logging into ECR"; tput sgr0;
	aws ecr get-login-password | docker login --username AWS --password-stdin 593453040104.dkr.ecr.eu-central-1.amazonaws.com

# Obtain AWS credentials
	tput setaf 3; echo "~ Obtaining AWS access keys"; tput sgr0
	ACCESS_KEY=$$(aws configure get default.aws_access_key_id)
	SECRET_KEY=$$(aws configure get default.aws_secret_access_key)
	SESSION_TOKEN=$$(aws configure get default.aws_session_token)
	tput setaf 2; echo "Success."; tput sgr0

# Build the container with credentials
	tput setaf 3; echo "~ Building docker image with tag: $(name)"; tput sgr0
	docker build \
		-t 593453040104.dkr.ecr.eu-central-1.amazonaws.com/$(name) \
		--build-arg AWS_ACCESS_KEY_ID=$$ACCESS_KEY \
		--build-arg AWS_SECRET_ACCESS_KEY=$$SECRET_KEY \
		--build-arg AWS_SESSION_TOKEN=$$SESSION_TOKEN \
		.
	@tput setaf 2; tput bold; echo "~ Container successfully built."; tput sgr0;

	docker push \
		593453040104.dkr.ecr.eu-central-1.amazonaws.com/$(name)

.ONESHELL:
submodule:
	@set -e
# Update submodule, fetch latest changes
	git submodule init
	git submodule update
