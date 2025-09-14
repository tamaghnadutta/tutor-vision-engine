# Monitoring Directory

This directory contains monitoring and observability scripts.

## Scripts:

- **test_api_with_monitoring.py** - API testing with monitoring integration

## Usage:

Run from project root:

```bash
make test-monitoring  # Generate API traffic for monitoring dashboard
```

This script works with the Prometheus/Grafana monitoring stack:

```bash
make monitoring-up    # Start monitoring stack
make test-monitoring  # Generate test traffic
```

Access dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090