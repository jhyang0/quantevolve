# Contributing to QuantEvolve

Thank you for your interest in contributing to QuantEvolve! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. Fork the repository.
2. Clone your fork: `git clone <your_repository_url_here> quantevolve` (Replace `<your_repository_url_here>` with the actual URL of this repository).
3. Navigate to the cloned directory: `cd quantevolve`.
4. Install the package in development mode: `pip install -e .`.
5. Run the tests to ensure everything is working: `python -m unittest discover tests` (Note: Test paths and commands might need updates based on project restructuring).

## Development Environment

We recommend using a virtual environment for development:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -e ".[dev]"
```

## Code Style

We follow the [Black](https://black.readthedocs.io/) code style. Please format your code before submitting a pull request:

```bash
black quantevolve tests
```

## Pull Request Process

1. Create a new branch for your feature or bugfix: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for your changes
4. Run the tests to make sure everything passes: `python -m unittest discover tests`
5. Commit your changes: `git commit -m "Add your descriptive commit message"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Submit a pull request to the main repository

## Adding Strategies and Modules

We encourage adding new strategies and modules to showcase QuantEvolve's capabilities, particularly in the domain of quantitative trading or other relevant financial applications. To add a new component:

1. Create appropriate modules within the `quantevolve` package structure (e.g., `quantevolve/strategy/` for new trading strategies or `quantevolve/indicators/` for new technical indicators).
2. Include all necessary files (e.g., initial strategy script, evaluation logic, sample data or data collection scripts, configuration files).
3. Add a `README.md` explaining the example, its purpose, and how to run it.
4. Ensure the example can be run with minimal setup, ideally leveraging the main `quantevolve-run.py` script and project's configuration system.

## Reporting Issues

When reporting issues, please include:

1. A clear description of the issue
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Python version, etc.)

## Feature Requests

Feature requests are welcome, especially those that enhance QuantEvolve's capabilities for trading strategy development. Please provide:

1. A clear description of the feature.
2. The motivation for adding this feature, particularly how it benefits quantitative trading strategy evolution.
3. Possible implementation ideas (if any).

## Code of Conduct

Please be respectful and considerate of others when contributing to the project. We aim to create a welcoming and inclusive environment for all contributors.
