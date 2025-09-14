# Scripts Directory

This directory contains the main application scripts used by the Makefile commands.

## Scripts:

- **analytics_dashboard.py** - Analytics and reporting dashboard
- **create_dataset.py** - Dataset generation from sample images
- **demo.py** - Demonstration script for error detection approaches
- **load_test.py** - Async load testing script
- **locustfile.py** - Locust-based load testing
- **run_eval.py** - Comprehensive evaluation runner

## Usage:

All scripts are designed to be run from the project root directory using the Makefile commands:

```bash
make eval          # Run comprehensive evaluation
make demo          # Demo all approaches
make load-test     # Run load testing
make locust-basic  # Run Locust load testing
```

See `make help` for all available commands.