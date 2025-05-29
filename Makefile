# Variables
PROJECT_DIR := $(shell pwd)
DOCKER_IMAGE := quantevolve
VENV_DIR := $(PROJECT_DIR)/env
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all            - Install dependencies and run tests"
	@echo "  venv           - Create a virtual environment"
	@echo "  install        - Install Python dependencies"
	@echo "  lint           - Run Black code formatting"
	@echo "  test           - Run tests"
	@echo "  docker-build   - Build the Docker image"
	@echo "  docker-run     - Run the Docker container with the example"

.PHONY: all
all: install test

# Create and activate the virtual environment
.PHONY: venv
venv:
	python3 -m venv $(VENV_DIR)

# Install Python dependencies in the virtual environment
.PHONY: install
install: venv
	$(PIP) install -e .

# Run Black code formatting
.PHONY: lint
lint: venv
	$(PYTHON) -m black quantevolve tests

# Run tests using the virtual environment
.PHONY: test
test: venv
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

# Build the Docker image
.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE) .

# Run the Docker container with the example
.PHONY: docker-run
docker-run:
	docker run --rm -v $(PROJECT_DIR):/app $(DOCKER_IMAGE) quantevolve/strategy/initial_strategy.py quantevolve/evaluation/quant_evaluator.py --config configs/quantevolve_config.yaml --output_dir quantevolve_output