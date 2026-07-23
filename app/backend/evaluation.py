# backend/evaluation.py
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
)

load_dotenv()


def run_nvidia_evaluations(model_id: str) -> pd.DataFrame:
    """1.

    Queries the selected Groq model for responses.
    2. Uses Ragas SingleTurnSample to evaluate Nvidia metrics without parameter mismatch errors.
    3. Compiles the scores into a Pandas DataFrame.
    """
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )

    # Bind selected Groq model as the LLM Judge
    llm = llm_factory(model_id, client=client)

    # Benchmark test cases
    benchmark_cases = [
        {
            "metric_name": "Answer Accuracy",
            "scorer": AnswerAccuracy(llm=llm),
            "user_input": "When was Einstein born?",
            "reference": "Albert Einstein was born in 1879.",
            "retrieved_contexts": ["Albert Einstein was born in 1879."],
        },
        {
            "metric_name": "Context Relevance",
            "scorer": ContextRelevance(llm=llm),
            "user_input": "When and where was Albert Einstein born?",
            "reference": "Albert Einstein was born in 1879 in Ulm, Germany.",
            "retrieved_contexts": [
                "Albert Einstein was born March 14, 1879.",
                "Albert Einstein was born at Ulm, in Württemberg, Germany.",
            ],
        },
        {
            "metric_name": "Response Groundedness",
            "scorer": ResponseGroundedness(llm=llm),
            "user_input": "Where was Albert Einstein born?",
            "reference": "Albert Einstein was born in Ulm, Germany.",
            "retrieved_contexts": [
                "Albert Einstein was born March 14, 1879.",
                "Albert Einstein was born at Ulm, in Württemberg, Germany.",
            ],
        },
    ]

    records = []

    for test in benchmark_cases:
        metric_name = test["metric_name"]
        scorer = test["scorer"]
        user_input = test["user_input"]
        reference = test["reference"]
        contexts = test["retrieved_contexts"]

        try:
            # Step 1: Query the model for an actual generated response
            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": user_input}],
            )
            generated_response = completion.choices[0].message.content

            # Step 2: Create a SingleTurnSample wrapper
            sample = SingleTurnSample(
                user_input=user_input,
                response=generated_response,
                reference=reference,
                retrieved_contexts=contexts,
            )

            # Step 3: Run synchronous scoring
            # Pass individual properties required by the specific scorer
            if metric_name == "Answer Accuracy":
                result = scorer.score(
                    user_input=sample.user_input,
                    response=sample.response,
                    reference=sample.reference,
                )
            elif metric_name == "Context Relevance":
                result = scorer.score(
                    user_input=sample.user_input,
                    retrieved_contexts=sample.retrieved_contexts,
                )
            elif metric_name == "Response Groundedness":
                result = scorer.score(
                    response=sample.response,
                    retrieved_contexts=sample.retrieved_contexts,
                )

            # Extract numeric value
            score_val = result.value if hasattr(result, "value") else float(result)

            records.append(
                {
                    "Metric": metric_name,
                    "User Input": user_input,
                    "Generated Response": generated_response,
                    "Reference / Context": reference,
                    "Score": round(float(score_val), 4),
                }
            )

        except Exception as e:
            records.append(
                {
                    "Metric": metric_name,
                    "User Input": user_input,
                    "Generated Response": f"Error: {str(e)}",
                    "Reference / Context": reference,
                    "Score": "N/A",
                }
            )

    return pd.DataFrame(records)
