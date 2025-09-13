from setuptools import setup, find_packages

setup(
    name="error-detection-api",
    version="0.1.0",
    description="AI-powered educational platform API for error detection in handwritten math solutions",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "openai>=1.3.7",
        "pillow>=10.1.0",
        "numpy>=1.24.3",
        "sqlalchemy>=2.0.23",
        "structlog>=23.2.0",
        "httpx>=0.25.2",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "ruff>=0.1.6",
            "mypy>=1.7.1",
            "isort>=5.12.0",
        ],
        "ocr": [
            "pytesseract>=0.3.10",
            "azure-cognitiveservices-vision-computervision>=0.9.0",
            "google-cloud-vision>=3.4.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "error-detection-api=src.api.main:main",
        ],
    },
)