# Testing Directory

This directory contains various test scripts for development and debugging.

## Test Scripts:

- **test_complete_solution_areas.py** - Tests complete solution area detection
- **test_endpoint_with_gpt4v_bounding_boxes.py** - GPT-4o endpoint testing
- **test_image_optimization.py** - Image processing optimization tests
- **test_model_router.py** - Model routing tests
- **test_parallel_optimization.py** - Parallel processing tests
- **test_single_sample.py** - Single sample testing
- **test_token_extraction.py** - Token extraction tests
- **test_token_tracking_integration.py** - Token tracking integration tests
- **quick_api_confidence_test.py** - Quick API confidence testing

## Usage:

Run from project root:

```bash
make test-complete-areas  # Test complete solution areas
python testing/test_single_sample.py  # Run individual tests
```