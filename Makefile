.PHONY: help requirements install install-all qc test ruff mypy prune-branches dataset
default: help
MAKEFLAGS += --no-print-directory

PACKAGE_DIR=geolife_clef_2024
SRC_FILES=${PACKAGE_DIR} tests

REQUIREMENTS_SUFFIX=$(shell [ -z ${extras} ] || echo '-${extras}')
REQUIREMENTS_MD5_FILE=$(shell [ -z ${extras} ] && echo 'requirements.in.md5' || echo 'pyproject.toml.${extras}.md5')
REQUIREMENTS_FILE=requirements${REQUIREMENTS_SUFFIX}.txt

pip-args: # Echo pip args
	@(command -v nvcc > /dev/null && \
	(python3 -c 'import requests' > /dev/null || pip install requests > /dev/null ) && \
	(python3 -c 'import packaging' > /dev/null || pip install packaging > /dev/null ) && \
	python3 -c 'import subprocess, requests, re, packaging.version;cuda_version = int(next(result for result in re.findall( r"(?<=release )(\d+)|(^-1)",subprocess.check_output(r"nvcc -V || echo -1", shell=True).decode(),flags=re.MULTILINE,)[0] if result));print("" if cuda_version == -1 else "--extra-index-url=https://download.pytorch.org/whl/"+(sorted([version for version in re.findall(r"^\<a href=\"(cpu|cu\d+)\/torch-(\d+\.\d+\.\d+)",requests.get("https://download.pytorch.org/whl/torch_stable.html").text,re.MULTILINE) if version[0].startswith("cu") if int(version[0][2:][:-1]) == cuda_version ], key=lambda x: (int(x[0][2:][-1]), packaging.version.Version(x[1])),reverse=True)[0][0]))' ) || echo ''

requirements: # Compile the pinned requirements if they've changed.
	@[ -f "${REQUIREMENTS_MD5_FILE}" ] && md5sum --status -c ${REQUIREMENTS_MD5_FILE} ||\
	( md5sum requirements.in $(shell [ -z ${extras} ] || echo pyproject.toml) > ${REQUIREMENTS_MD5_FILE} && rm -rf ${REQUIREMENTS_FILE} );\
	[ ! -f "${REQUIREMENTS_FILE}" ] && (python3 -c 'import piptools' || pip install pip-tools ) && pip-compile --no-emit-index-url $(shell echo '${REQUIREMENTS_MD5_FILE}' | grep -oP '^([^\.]*?\.)[^\.]*' ) $(shell [ -z ${extras} ] || echo '--extra ${extras}' ) $(shell make pip-args) -o ${REQUIREMENTS_FILE} 

requirements: extras=

install: # Install minimum required packages.
	@make requirements && pip install -e .${extras} $(shell make pip-args) --upgrade

install-all: # Install all packages
	@make requirements; make requirements extras=all && pip install -e .[all] $(shell make pip-args) --upgrade

ruff: # Run ruff
	@ruff check ${SRC_FILES} --fix

mypy: # Run mypy
	mypy ${SRC_FILES}

test: # Run pytest
	@pytest --cov=${PACKAGE_DIR} tests --cov-report term-missing

qc:  # Format and test
	@make ruff; make mypy; make test

prune-branches: # Remove all branches except one
	@git branch | grep -v "${except}" | xargs git branch -D

prune-branches: except=main

dataset: # Download the dataset
	@[ -f "./data/GLC24_P0_metadata_train.csv" ] || ((python3 -c 'import kaggle' || python3 -m pip install kaggle) && kaggle competitions download -c geolifeclef-2024 && unzip -o geolifeclef-2024.zip -d ./data && rm -rf geolifeclef-2024.zip)

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m\n\t$$(echo $$l | cut -f 2- -d'#')\n"; done
