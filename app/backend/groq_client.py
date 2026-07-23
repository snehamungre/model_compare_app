from functools import lru_cache
import requests


@lru_cache(maxsize=1)
def get_models(api_key: str):
    """Fetch available Groq models and pricing details."""

    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    model_ids = []
    model_dict = {}

    # For each model get relevent information
    for m in data.get("data", []):
        m_id = m["id"]
        model_ids.append(m_id)

        pricing = m.get("pricing", {})
        model_dict[m_id] = {
            "name": m.get("name"),
            "context_window": m.get("context_window"),
            "max_output": m.get("max_completion_tokens"),
            "prompt_cost_per_token": float(pricing.get("prompt", 0)),
            "completion_cost_per_token": float(pricing.get("completion", 0)),
        }

    return model_ids, model_dict
