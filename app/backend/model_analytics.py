# Calculate usage metrics and returns formatted string response
def calculate_cost_and_metrics(usage, model_id, model_dict):
    if not usage:
        return "Usage statistics unavailable.", {}

    # Calculate metrics
    model_info = model_dict.get(model_id, {})
    p_rate = model_info.get("prompt_cost_per_token", 0.0)
    c_rate = model_info.get("completion_cost_per_token", 0.0)

    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    prompt_cost = prompt_tokens * p_rate
    completion_cost = completion_tokens * c_rate
    total_cost = prompt_cost + completion_cost

    latency = getattr(usage, "total_time", None)

    # Record in dictionary for summary
    record = {
        "Model": model_id,
        "Prompt Tokens": prompt_tokens,
        "Completion Tokens": completion_tokens,
        "Total Tokens": total_tokens,
        "Latency (s)": round(latency, 3) if latency else "N/A",
        "Cost ($)": f"${total_cost:.6f}",
    }

    # Format metrics to display in chat
    analysis_lines = [
        f"**Model:** `{model_id}`",
        f"**Tokens Used:** {total_tokens:,} ({prompt_tokens:,} prompt + {completion_tokens:,} completion)",
    ]

    if latency:
        analysis_lines.append(f"**Latency:** {latency:.2f}s")

    analysis_lines.append(f"**Estimated Cost:** `${total_cost:.6f}` USD")

    return "\n\n".join(analysis_lines), record


from openai import OpenAI


def generate_model_response(model_id: str, prompt: str) -> str:
    """Sends a user prompt to the selected Groq model and returns its response text."""
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )

    completion = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )

    return completion.choices[0].message.content
