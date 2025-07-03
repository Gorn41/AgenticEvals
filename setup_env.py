#!/usr/bin/env python3
"""
Environment setup script for LLM-AgentTypeEval.

This script helps users set up their API keys and configuration.
"""

import os
import sys
from pathlib import Path


def create_env_file():
    """Create .env file from template."""
    env_file = Path(".env")
    example_file = Path(".env.example")
    
    if env_file.exists():
        response = input(".env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return False
    
    if not example_file.exists():
        print("Error: .env.example file not found!")
        return False
    
    # Copy example to .env
    content = example_file.read_text()
    env_file.write_text(content)
    print(f"Created {env_file}")
    return True


def get_api_key():
    """Prompt user for API key."""
    print("\nTo use this system, you need a Gemini API key.")
    print("Get your key from: https://makersuite.google.com/app/apikey")
    print()
    
    api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("No API key provided. You can add it later to the .env file.")
        return None
    
    return api_key


def update_env_file(api_key: str):
    """Update .env file with the provided API key."""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("Error: .env file not found!")
        return False
    
    # Read current content
    lines = env_file.read_text().splitlines()
    
    # Update the API key line
    updated_lines = []
    key_updated = False
    
    for line in lines:
        if line.startswith("GOOGLE_API_KEY="):
            updated_lines.append(f"GOOGLE_API_KEY={api_key}")
            key_updated = True
        else:
            updated_lines.append(line)
    
    # If no existing key line found, add it
    if not key_updated:
        updated_lines.append(f"GOOGLE_API_KEY={api_key}")
    
    # Write back to file
    env_file.write_text("\n".join(updated_lines) + "\n")
    print(f"Updated {env_file} with your Gemini API key.")
    return True


def test_setup():
    """Test if the setup works by trying to load the configuration."""
    try:
        # Add src to path for testing
        sys.path.insert(0, str(Path("src")))
        
        from utils.config import get_config_manager
        
        config_manager = get_config_manager()
        api_key = config_manager.config.api_keys.get("google")
        
        if api_key and api_key != "your_google_api_key_here":
            print("Configuration loaded successfully!")
            print(f"Gemini API key found: {api_key[:8]}...")
            return True
        else:
            print("No valid Gemini API key found in configuration.")
            print("Make sure you've set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
            return False
            
    except Exception as e:
        print(f"Error testing configuration: {e}")
        return False


def main():
    """Main setup routine."""
    print("LLM-AgentTypeEval Environment Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("Error: This script must be run from the project root directory.")
        print("Make sure you can see the 'src' folder in the current directory.")
        return 1
    
    # Step 1: Create .env file
    print("\nStep 1: Setting up environment file...")
    if not create_env_file():
        return 1
    
    # Step 2: Get API key
    print("\nStep 2: API Key Configuration...")
    api_key = get_api_key()
    
    if api_key:
        if not update_env_file(api_key):
            return 1
    
    # Step 3: Test setup
    print("\nStep 3: Testing configuration...")
    if test_setup():
        print("\nSetup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python3 quick_start.py' to test the system")
        print("2. Run 'python3 example_usage.py' for a full demonstration")
        print("3. Run 'python3 run_tests.py --unit' to run the test suite")
    else:
        print("\nSetup completed with warnings.")
        print("Check the .env file and make sure your Gemini API key is correct.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 