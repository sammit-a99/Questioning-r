import json
import requests
import csv
from datetime import datetime


def get_provider_from_id(model_id):
    """
    Extracts the provider string directly from the model ID prefix
    and returns a cleanly capitalized title format.
    e.g., 'google/gemini-2.5-flash' -> 'Google'
          'meta-llama/llama-3-70b'  -> 'Meta Llama'
    """
    if not model_id or "/" not in model_id:
        return "Unknown"

    # OpenRouter model IDs always match 'provider-slug/model-slug'
    raw_provider = model_id.split('/')[0]

    # Replace dashes with spaces and apply Title Case capitalization
    return raw_provider.replace('-', ' ').title()


def fetch_openrouter_pricing():
    # Public endpoint to retrieve all models and properties
    url = "https://openrouter.ai/api/v1/models"

    print("Fetching available models from OpenRouter...")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data: {e}")
        return

    models = payload.get("data", [])
    if not models:
        print("No models found in the API response.")
        return

    print(f"Successfully retrieved {len(models)} models.\n")

    # Write to CSV file
    with open('openrouter_models.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model Name', 'Model ID', 'Provider', 'Input ($/1M)', 'Output ($/1M)', 'Context Window', 'Release Date'])

        for model in models:
            model_name = model.get("name", "Unknown")
            model_id = model.get("id", "Unknown")

            created_timestamp = model.get("created")
            if created_timestamp:
                try:
                    release_date = datetime.fromtimestamp(created_timestamp).strftime('%Y-%m-%d')
                except (ValueError, TypeError, OverflowError):
                    release_date = "Unknown"
            else:
                release_date = "N/A"


            context_length = model.get("context_length", "N/A")

            # OpenRouter pricing returns cost per individual token natively
            pricing = model.get("pricing", {})
            prompt_per_token = pricing.get("prompt", "0")
            completion_per_token = pricing.get("completion", "0")

            # Convert cost per token to cost per 1 Million tokens for readable values
            try:
                input_per_m = float(prompt_per_token) * 1_000_000
                input_str = f"${input_per_m:,.4f}" if input_per_m > 0 else "Free"
            except (ValueError, TypeError):
                input_str = "N/A"

            try:
                output_per_m = float(completion_per_token) * 1_000_000
                output_str = f"${output_per_m:,.4f}" if output_per_m > 0 else "Free"
            except (ValueError, TypeError):
                output_str = "N/A"

            # Format context length for display
            context_str = f"{context_length:,}" if isinstance(context_length, int) else context_length

            # Extract provider from model ID
            provider = get_provider_from_id(model_id)

            writer.writerow([model_name, model_id, provider, input_str, output_str, context_str, release_date])

    print("Data written to openrouter_models.csv")


if __name__ == "__main__":
    fetch_openrouter_pricing()
