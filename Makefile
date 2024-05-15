export PATH := /app/.local/bin:$(PATH)
export REPO := https://github.com/joaquinbejar/MarketMimic.git
VENV_NAME?=venv
VENV_ACTIVATE=$(VENV_NAME)/bin/activate
PYTHON_PATH=$(shell which python3.11)

create-venv: delete-venv
	$(PYTHON_PATH) -m venv $(VENV_NAME)
	@echo "To activate the venv, run 'source $(VENV_ACTIVATE)'"

delete-venv:
	rm -rf venv

uninstall:
	pip uninstall -y marketmimic

clean-build-and-dist:
	rm -rf ./build
	rm -rf ./dist
	rm -rf marketmimic.egg-info

build: clean-build-and-dist
	pip install build
	python -m build --wheel --sdist

clean :
	rm -rf build/
	rm -rf *.egg-info

reinstall-dependencies: update-pip delete-dependencies install-dep clean

update-pip:
	python -m pip install --upgrade pip --no-cache-dir
	python -m pip install pip~=23.2.1 --force-reinstall --no-cache-dir
	pip install --upgrade pip

delete-dependencies:
	pip freeze | xargs -I %v pip uninstall -y '%v'


### ***** UNIT TESTS ***** ###

run-unit-test-coverage:
	coverage run --source=marketmimic -m unittest discover -v -s ./tests/unit/ -p '*test*.py'
	coverage report
	coverage html -d coverage_html
	echo report at '$(COVERAGE_LOCATION)'

run-unit-tests:		## Run unit tests
	python -m unittest discover -v -s ./tests/unit/ -p '*test*.py'

run-extended-tests:		## Run extended tests
	python -m unittest discover -v -s ./tests/extended/ -p '*test*.py'


### ***** CI PIPELINE ***** ###

install-dep:	## Install dependencies
	pip install .

test:		## Run tests
	pip install pytest mockito coverage freezegun pytest-cov
	pip install .[tests]
	coverage run --source=src -m pytest tests/unit/ && coverage report -m
