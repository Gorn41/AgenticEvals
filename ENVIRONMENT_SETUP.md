# Environment Setup Guide

This guide explains how to set up your API keys and environment configuration for AgenticEvals.

## Quick Setup

The easiest way to set up your environment is using the interactive setup script:

```bash
python3 setup_env.py
```

This script will:
1. Create a `.env` file from the template
2. Prompt you for your Gemini API key
3. Optionally prompt you for a Hugging Face Hub token (for local vLLM models)
4. Configure the Selenium MCP endpoint (used by a web benchmark)
5. Test the configuration
6. Provide troubleshooting help if needed

## Manual Setup

### 1. Create Environment File

Copy the example environment file:
```bash
cp .env.example .env
```

### 2. Configure API Keys

Edit the `.env` file and replace the placeholder with your actual API key:

```bash
# .env
GOOGLE_API_KEY=your_gemini_api_key_here
HUGGING_FACE_HUB_TOKEN=your_hf_token_if_needed
```

Or use the alternative key name:
```bash
GOOGLE_API_KEY=your_actual_gemini_api_key
```

### 3. Get Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the generated key
4. Paste it into your `.env` file

### 4. Optional: Get a Hugging Face Hub Token

If you plan to use local vLLM with gated models, create a token at: https://huggingface.co/settings/tokens and set `HUGGING_FACE_HUB_TOKEN` in your `.env`.

### 5. Test Your Setup

```bash
python3 quick_start.py
```

## Environment Variables

### Required

- `GOOGLE_API_KEY`: Your Gemini API key (primary)
- `GEMINI_API_KEY`: Alternative name for the same key

### Optional

- `AGENTIC_EVALS_LOG_LEVEL`: Set to `DEBUG`, `INFO`, `WARNING`, or `ERROR`
- `AGENTIC_EVALS_CONFIG_FILE`: Path to custom configuration file
- `HUGGING_FACE_HUB_TOKEN`: Your Hugging Face access token (for gated models via vLLM)

## Shell-Specific Setup

### Bash/Zsh
```bash
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

### Fish Shell
```fish
set -gx GOOGLE_API_KEY "your_gemini_api_key_here"
```

### Windows Command Prompt
```cmd
set GOOGLE_API_KEY=your_gemini_api_key_here
```

### Windows PowerShell
```powershell
$env:GOOGLE_API_KEY="your_gemini_api_key_here"
```

## Configuration File (Advanced)

You can also use a YAML configuration file:

```yaml
# config.yaml
models:
  google: "your_gemini_api_key_here"

benchmarks:
  timeout_seconds: 30
  max_retries: 3
  collect_detailed_metrics: true
  save_responses: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

Load with:
```python
from utils.config import get_config_manager

config = get_config_manager("config.yaml")
```

## Troubleshooting

### Common Issues

1. **"No API key found"**: 
   - Make sure your `.env` file exists and contains `GOOGLE_API_KEY=your_key`
   - Check that there are no extra spaces around the `=`
   - Verify the API key format: `GOOGLE_API_KEY=your_gemini_key_here` (no spaces around =)

2. **"API key invalid"**: 
   - Verify your key is correct at [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Make sure you're using a Gemini API key, not a different Google service key
   - Try setting the environment variable directly: `export GOOGLE_API_KEY="your_gemini_key"`

3. **"Import errors"**: 
   - Run from the project root directory
   - Make sure you've installed dependencies: `pip install -r requirements.txt`
   - Check your Python path includes the `src` directory

4. **"Permission denied"**: 
   - Check file permissions on your `.env` file
   - Make sure the script is executable: `chmod +x setup_env.py`

### Debug Mode

Run with debug output to see detailed information:

```bash
AGENTIC_EVALS_LOG_LEVEL=DEBUG python3 quick_start.py
```

### Manual Verification

Test your API key manually:

```python
import os
from models.loader import load_gemini

api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key found: {api_key[:8]}..." if api_key else "No API key found")

model = load_gemini("gemini-2.5-flash", api_key=api_key)
response = await model.generate("Test prompt")
print(f"Model response: {response.text}")
```

## Security Best Practices

1. **Never commit your `.env` file** to version control
2. **Use separate API keys** for development and production
3. **Rotate your API keys** regularly
4. **Monitor usage** in the Google AI Studio console
5. **Set up API quotas** to prevent unexpected charges

## Getting Help

If you're still having issues:

1. Check the [troubleshooting section](TESTING.md#troubleshooting) in TESTING.md
2. Run the diagnostic script: `python3 setup_env.py`
3. Open an issue on GitHub with your error message and setup details
4. Join our Discord/Slack for real-time help

## Next Steps

Once your environment is set up:

1. **Run the quick start**: `python3 quick_start.py`
2. **Try the full example**: `python3 example_usage.py`
3. **Run the tests**: `python3 run_tests.py --unit`
4. **Explore the benchmarks**: Check out `src/benchmarks/` for available tests 