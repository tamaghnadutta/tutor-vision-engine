# Error Detection API

An AI-powered educational platform API that provides real-time feedback on student handwritten mathematical solutions.

## Overview

This API analyzes student work and identifies step-level errors in mathematical solutions, providing helpful feedback including error descriptions, corrections, and hints.

## Features

- **Gemini 2.5 Flash Integration**: Unified AI processing using Google's latest model
- **Structured Pydantic Outputs**: Type-safe, validated responses throughout
- **Error Detection**: Step-level error identification in handwritten math solutions
- **Real-time Feedback**: Provides corrections and hints for identified errors
- **Vision + OCR + Reasoning**: All-in-one AI pipeline for comprehensive analysis
- **Concurrent Processing**: Handles ≥5 concurrent requests with p95 ≤ 10s latency
- **Observability**: Structured logging and metrics
- **Persistence**: Request/response auditing
- **Security**: API key authentication

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key (or other LLM provider)
- Optional: OCR service credentials

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd error-detection-api

# Install dependencies
make setup

# Configure environment
cp .env.sample .env
# Edit .env with your API keys

# Run the demo
make demo

# Run evaluation
make eval
```

## API Usage

### Endpoint

`POST /detect-error`

### Request

```json
{
  "question_url": "https://example.com/question_image.png",
  "solution_url": "https://example.com/solution_image.png",
  "bounding_box": {
    "minX": 316,
    "maxX": 635,
    "minY": 48.140625,
    "maxY": 79.140625
  },
  "user_id": "optional_user_identifier",
  "session_id": "optional_session_identifier",
  "question_id": "optional_question_identifier"
}
```

### Response

```json
{
  "job_id": "unique_job_identifier",
  "y": 150.5,
  "error": "Incorrect application of quadratic formula",
  "correction": "The discriminant should be b² - 4ac, not b² + 4ac",
  "hint": "Remember: discriminant = b² - 4ac for quadratic equations",
  "solution_complete": false,
  "contains_diagram": true,
  "question_has_diagram": true,
  "solution_has_diagram": false,
  "llm_used": true,
  "solution_lines": [],
  "llm_ocr_lines": []
}
```

## Development

### Project Structure

```
├── src/
│   ├── api/              # FastAPI application
│   ├── models/           # ML models and inference
│   ├── data/             # Data handling and preprocessing
│   ├── eval/             # Evaluation framework
│   ├── utils/            # Utilities and helpers
│   └── config/           # Configuration management
├── data/                 # Dataset and test data
│   ├── sample_images/    # Real sample images (Q1-Q4, Attempt1-4)
│   ├── eval_results/     # Evaluation results
│   └── processed/        # Processed data
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── docs/                 # Documentation
└── docker/               # Docker configuration
```

### Commands

- `make setup` - Install dependencies and setup environment
- `make dev` - Start development server
- `make test` - Run test suite
- `make eval` - Run evaluation harness
- `make demo` - Run demo with sample images
- `make create-dataset` - Generate dataset from provided images
- `make lint` - Run code linting
- `make format` - Format code

## Architecture

See [docs/Architecture.md](docs/Architecture.md) for detailed system architecture.

## Performance

- **Latency**: p50/p90/p95 metrics reported
- **Throughput**: Handles 5+ concurrent requests
- **Cost**: Estimated cost per 100 requests provided

## License

[License information]