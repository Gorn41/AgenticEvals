#!/usr/bin/env python3
"""
Environment setup script for AgenticEvals.

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


def get_hf_token():
    """Prompt user for Hugging Face Hub token (optional)."""
    print("\nOptional: Provide a Hugging Face Hub token for gated/open models on Hugging Face.")
    print("If you plan to run local models via vLLM and the model is gated, you'll need this token.")
    print("Create a token at: https://huggingface.co/settings/tokens")
    print()
    token = input("Enter your HUGGING_FACE_HUB_TOKEN (or press Enter to skip): ").strip()
    if not token:
        print("No HF token provided. You can add it later to the .env file.")
        return None
    return token

def get_mcp_endpoint():
    """Prompt user for a local Selenium MCP WebSocket URL."""
    print("\nOptional: Configure local Selenium MCP endpoint (for web navigation benchmark).")
    print("If you run a local MCP server, enter its WebSocket URL (e.g., ws://127.0.0.1:7007).")
    print("Otherwise, press Enter to skip and configure later in .env.")
    mcp_url = input("Enter SELENIUM_MCP_URL (or press Enter to skip): ").strip()
    return mcp_url or None


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


def update_env_with_hf_token(token: str) -> bool:
    """Add or update HUGGING_FACE_HUB_TOKEN in .env."""
    env_file = Path(".env")
    if not env_file.exists():
        print("Error: .env file not found!")
        return False
    lines = env_file.read_text().splitlines()
    updated_lines = []
    updated = False
    for line in lines:
        if line.startswith("HUGGING_FACE_HUB_TOKEN="):
            updated_lines.append(f"HUGGING_FACE_HUB_TOKEN={token}")
            updated = True
        else:
            updated_lines.append(line)
    if not updated:
        updated_lines.append(f"HUGGING_FACE_HUB_TOKEN={token}")
    env_file.write_text("\n".join(updated_lines) + "\n")
    print(f"Updated {env_file} with HUGGING_FACE_HUB_TOKEN.")
    return True


def update_env_with_mcp(url: str) -> bool:
    """Add or update SELENIUM_MCP_URL in .env."""
    from pathlib import Path
    env_file = Path(".env")
    if not env_file.exists():
        print("Error: .env file not found!")
        return False
    lines = env_file.read_text().splitlines()
    updated_lines = []
    updated = False
    for line in lines:
        if line.startswith("SELENIUM_MCP_URL="):
            updated_lines.append(f"SELENIUM_MCP_URL={url}")
            updated = True
        else:
            updated_lines.append(line)
    if not updated:
        updated_lines.append(f"SELENIUM_MCP_URL={url}")
    env_file.write_text("\n".join(updated_lines) + "\n")
    print(f"Updated {env_file} with SELENIUM_MCP_URL.")
    return True


def test_setup():
    """Test if the setup works by trying to load the configuration."""
    try:
        # Add src to path for testing
        sys.path.insert(0, str(Path("src").resolve()))
        
        from src.utils.config import get_config_manager
        
        config_manager = get_config_manager()
        api_key = config_manager.config.api_keys.get("google")
        
        if api_key and len(api_key.strip()) > 10 and not api_key.startswith("your_"):
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
    print("AgenticEvals Environment Setup")
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
    
    # Step 2: API Key Configuration (Gemini)
    print("\nStep 2: API Key Configuration (Gemini)...")
    api_key = get_api_key()
    
    if api_key:
        if not update_env_file(api_key):
            return 1
    
    # Step 3: Hugging Face Token (optional)
    print("\nStep 3: Hugging Face Token (optional for vLLM)...")
    hf_token = get_hf_token()
    if hf_token:
        update_env_with_hf_token(hf_token)

    # Step 4: MCP Endpoint
    print("\nStep 4: Local Selenium MCP Endpoint...")
    mcp_url = get_mcp_endpoint()
    if mcp_url:
        update_env_with_mcp(mcp_url)
    else:
        # Default to localhost if not provided
        update_env_with_mcp("ws://127.0.0.1:7007")

    # Step 5: Test setup
    print("\nStep 5: Testing configuration...")
    if test_setup():
        print("\nSetup completed successfully!")
    else:
        print("\nSetup completed with warnings.")
        print("Check the .env file and make sure your Gemini API key is correct.")
    print("\nNote: To run the local web navigation benchmark, you must start a Selenium MCP server listening at SELENIUM_MCP_URL. If you don't have one, see README for instructions.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 