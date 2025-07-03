"""
Setup script for AgenticEvals.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agentic-evals",
    version="0.1.0",
    description="A comprehensive benchmark for evaluating LLMs across classic AI agent types",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nattaput (Gorn) Namchittai",
    author_email="gorn41@outlook.com",
    url="https://github.com/Gorn41/AgenticEvals",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentic-evals=cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords=[
        "llm", "language-models", "ai-agents", "benchmarking", 
        "evaluation", "artificial-intelligence", "machine-learning"
    ],
    project_urls={
        "Bug Reports": "https://github.com/Gorn41/AgenticEvals/issues",
        "Source": "https://github.com/Gorn41/AgenticEvals",
    },
)