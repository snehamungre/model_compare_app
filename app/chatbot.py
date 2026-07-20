import requests
import json
from functools import lru_cache


@lru_cache(maxsize=1)
def get_models(api_key: str):
    # Using the groq API to retrive all the models avalible
    url = "https://api.groq.com/openai/v1/models"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    response = requests.get(url, headers=headers)
    data = response.json()

    model_ids = []
    model_dict = {}

    # parsing json to return the model id and model specifications
    for m in data["data"]:
        m_id = m["id"]
        model_ids.append(m_id)

        pricing = m.get("pricing", {})

        model_dict[m_id] = {
            "name": m.get("name"),
            "context_window": m.get("context_window"),
            "max_output": m.get("max_completion_tokens"),
            # Storing rates as floats for direct calculation
            "prompt_cost_per_token": float(pricing.get("prompt", 0)),
            "completion_cost_per_token": float(pricing.get("completion", 0)),
        }

    return model_ids, model_dict


def calculate_cost_and_metrics(usage, model_id, model_dict):
    if not usage:
        return "Usage statistics unavailable."

    # Retrieve per-token rates from model_dict
    model_info = model_dict.get(model_id, {})
    p_rate = model_info.get("prompt_cost_per_token", 0.0)
    c_rate = model_info.get("completion_cost_per_token", 0.0)

    # Calculate individual and total costs
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    prompt_cost = prompt_tokens * p_rate
    completion_cost = completion_tokens * c_rate
    total_cost = prompt_cost + completion_cost

    # Extract optional details like reasoning tokens (for models like Qwen or GPT-OSS)
    reasoning_tokens = 0
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning_tokens = getattr(
            usage.completion_tokens_details, "reasoning_tokens", 0
        )

    # Format the metrics breakdown string
    analysis_lines = [
        f"**Model:** `{model_id}`",
        f"**Tokens Used:** {total_tokens:,} ({prompt_tokens:,} prompt + {completion_tokens:,} completion)",
    ]

    if reasoning_tokens > 0:
        analysis_lines.append(
            f"**Reasoning Tokens:** {reasoning_tokens:,} included in completion"
        )

    if hasattr(usage, "total_time") and usage.total_time:
        analysis_lines.append(f"**Latency:** {usage.total_time:.2f}s")

    analysis_lines.append(f"**Estimated Cost:** `${total_cost:.6f}` USD")

    return "\n\n".join(analysis_lines)
