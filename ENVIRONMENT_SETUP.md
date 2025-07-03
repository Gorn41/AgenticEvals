# Environment Setup Guide

This guide explains how to set up your API keys and environment configuration for LLM-AgentTypeEval.

## Quick Setup (Recommended)

Use the automated setup script:

```bash
python3 setup_env.py
```

This script will:
1. Create a `.env` file from the template
2. Prompt you for your Google API key
3. Test the configuration
4. Guide you through next steps

## Manual Setup

### 1. Create Environment File

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Add Your API Key

Edit the `.env` file and replace the placeholder with your actual API key:

```bash
# Before
GOOGLE_API_KEY=your_gemini_api_key_here

# After
GOOGLE_API_KEY=your_actual_gemini_api_key
```

### 3. Get Your API Key

Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey):

1. Go to https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key
5. Paste it into your `.env` file

## Environment Variables

The system supports these environment variables:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_API_KEY` | Gemini API key (from Google AI Studio) | Yes | - |
| `GEMINI_API_KEY` | Alternative Gemini API key variable | No | - |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No | INFO |
| `TIMEOUT_SECONDS` | API request timeout | No | 60 |
| `RESULTS_DIR` | Directory for saving results | No | ./results |

## Security Notes

- **Never commit your `.env` file** to version control
- The `.env` file is automatically ignored by git (see `.gitignore`)
- Keep your API key secure and don't share it publicly
- Use environment variables in production deployments

## Alternative Methods

### Environment Variables

Instead of using a `.env` file, you can set environment variables directly:

```bash
# Bash/Zsh
export GOOGLE_API_KEY="your_gemini_api_key_here"

# Fish
set -gx GOOGLE_API_KEY "your_gemini_api_key_here"

# Windows Command Prompt
set GOOGLE_API_KEY=your_gemini_api_key_here

# Windows PowerShell
$env:GOOGLE_API_KEY="your_gemini_api_key_here"
```

### Configuration File

You can also create a YAML configuration file:

```yaml
# config.yaml
default_model: "gemini-1.5-pro"
api_keys:
  google: "your_gemini_api_key_here"
log_level: "INFO"
timeout_seconds: 60
```

Then load it in your code:
```python
from utils.config import ConfigManager

config_manager = ConfigManager(config_path="config.yaml")
```

## Testing Your Setup

After setting up your environment, test it with:

```bash
# Quick test
python3 quick_start.py

# Full example
python3 example_usage.py

# Run tests
python3 run_tests.py --unit
```

## Troubleshooting

### "No API key found" error

1. Check that your `.env` file exists in the project root
2. Verify the API key format: `GOOGLE_API_KEY=your_gemini_key_here` (no spaces around =)
3. Make sure the key is valid by testing it at https://makersuite.google.com/

### "Import error" when testing

1. Make sure you've installed dependencies: `pip install -r requirements.txt`
2. Check that you're in the project root directory
3. Verify the `src/` directory exists

### "Configuration not found" error

1. Run the setup script: `python3 setup_env.py`
2. Check file permissions on `.env` file
3. Try setting the environment variable directly: `export GOOGLE_API_KEY="your_gemini_key"`

## Production Deployment

For production environments:

1. **Use environment variables** instead of `.env` files
2. **Use secrets management** (AWS Secrets Manager, Azure Key Vault, etc.)
3. **Rotate API keys** regularly
4. **Monitor API usage** to detect unauthorized access
5. **Use least-privilege access** principles

Example Docker setup:
```dockerfile
# Dockerfile
ENV GOOGLE_API_KEY=""
# Set via docker run -e GOOGLE_API_KEY="your_key" ...
```

Example Kubernetes setup:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
type: Opaque
stringData:
  google-api-key: "your_key_here"
---
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: app
        env:
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: google-api-key
```

This ensures your API keys are properly secured and managed in production environments. 