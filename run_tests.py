#!/usr/bin/env python3
"""
Test runner script for AgenticEvals.

This script provides convenient ways to run different categories of tests.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description="", env=None):
    """Run a command and return the exit code."""
    if description:
        print(f"\n{description}")
        print("-" * len(description))
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, env=env)
    return result.returncode


def check_dependencies():
    """Check if test dependencies are installed."""
    try:
        import pytest
        import pytest_asyncio
        return True
    except ImportError:
        return False


def install_dependencies():
    """Install test dependencies."""
    dependencies = [
        "pytest", "pytest-asyncio", "pytest-cov"
    ]
    
    print("Installing test dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install"
        ] + dependencies, check=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run AgenticEvals tests")
    parser.add_argument(
        "--unit", "-u", 
        action="store_true", 
        help="Run only unit tests (fast, no external dependencies)"
    )
    parser.add_argument(
        "--integration", "-i", 
        action="store_true",
        help="Run integration tests (requires API keys)"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run tests with coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Run specific test file"
    )
    parser.add_argument(
        "--class", "-k",
        type=str,
        dest="test_class",
        help="Run specific test class"
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        help="Run specific test method"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("Error: tests directory not found. Run this script from the project root.")
        return 1
    
    # Install dependencies if requested
    if args.install_deps:
        return 0 if install_dependencies() else 1
    
    # Check dependencies
    if not check_dependencies():
        print("Test dependencies not found. Install with:")
        print("python run_tests.py --install-deps")
        print("# or manually:")
        print("pip install pytest pytest-asyncio pytest-cov")
        return 1
    
    # Set up environment for proper imports
    env = os.environ.copy()
    project_root = Path.cwd()
    
    # Add project root to PYTHONPATH so src package can be imported
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{project_root}:{pythonpath}"
    else:
        env["PYTHONPATH"] = str(project_root)
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", "--asyncio-mode=auto"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    # Handle test selection
    if args.unit:
        cmd.extend(["-m", "not integration and not slow"])
        description = "Running unit tests only"
    elif args.integration:
        cmd.extend(["-m", "integration"])
        description = "Running integration tests"
        
        # Check for API keys
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: No Gemini API key found for integration tests.")
            print("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
            print("Integration tests may be skipped.")
    else:
        description = "Running all tests"
    
    # Handle specific test selection
    if args.file:
        test_path = f"tests/{args.file}" if not args.file.startswith("tests/") else args.file
        cmd.append(test_path)
        description = f"Running tests from {test_path}"
        
        if args.test_class:
            cmd[-1] = f"{cmd[-1]}::{args.test_class}"
            description += f" (class: {args.test_class})"
            
            if args.method:
                cmd[-1] = f"{cmd[-1]}::{args.method}"
                description += f" (method: {args.method})"
    elif args.test_class or args.method:
        print("Error: --class and --method require --file")
        return 1
    
    # Run tests
    result = run_command(cmd, description, env=env)
    
    # Print coverage info if generated
    if args.coverage and result == 0:
        print("\nCoverage report generated in htmlcov/index.html")
    
    return result


if __name__ == "__main__":
    sys.exit(main()) 