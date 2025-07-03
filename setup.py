"""
Setup script for LLM-AgentTypeEval.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="llm-agenttype-eval",
    version="0.1.0",
    description="Benchmark for evaluating LLMs across five classic AI agent architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LLM-AgentTypeEval Team",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "logging": [
            "loguru>=0.7.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agenttype-eval=cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="llm evaluation benchmark ai agents",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-agenttype-eval/issues",
        "Source": "https://github.com/yourusername/llm-agenttype-eval",
    },
) 