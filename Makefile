.PHONY: help setup data test lint clean

PYTHON := python
PIP := pip
PYTEST := pytest
FLAKE8 := flake8

help:
	@echo "Available targets:"
	@echo "  setup    Install dependencies and the package in editable mode"
	@echo "  data     Download and extract the California Housing dataset"
	@echo "  test     Run the test suite"
	@echo "  lint     Lint source code with flake8"
	@echo "  clean    Remove build artefacts and caches"

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PYTHON) -c "from capstone.data import fetch_housing_data; fetch_housing_data()"

test:
	$(PYTEST) tests/

lint:
	$(FLAKE8) src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
