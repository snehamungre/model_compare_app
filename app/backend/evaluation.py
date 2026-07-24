# backend/evaluation.py
import asyncio
import inspect
import os
import traceback
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_groq import ChatGroq
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory, LangchainLLMWrapper
from ragas.metrics import (
    AspectCritic,
    FactualCorrectness,
    LLMContextRecall,
    ResponseRelevancy,
    SimpleCriteriaScore,
)
from ragas.metrics.collections import (
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

# ragas.embeddings.HuggingfaceEmbeddings is broken/abstract in current Ragas
# releases (missing aembed_documents/aembed_query — see ragas issue #1806).
# Wrapping the real langchain_huggingface embeddings class instead sidesteps
# that bug entirely and is the pattern Ragas's own docs recommend.
embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
)

load_dotenv()


async def evaluate_sample_async(scorer, sample: SingleTurnSample):
    """Universal async helper to score any Ragas or Nvidia metric.

    Returns a (score, error) tuple. `error` is None on success, or a short
    string describing what went wrong so failures are visible in the UI
    instead of silently showing up as a plausible-looking 0.0.
    """
    metric_name = getattr(scorer, "name", scorer.__class__.__name__)
    try:
        # 1. Try single_turn_ascore (Ragas standard single-turn metrics)
        if hasattr(scorer, "single_turn_ascore"):
            res = await scorer.single_turn_ascore(sample)
            if hasattr(res, "value"):
                return (float(res.value) if res.value is not None else 0.0), None
            return (float(res) if res is not None else 0.0), None

        # 2. Try ascore (Ragas metric collection / Nvidia metrics)
        elif hasattr(scorer, "ascore"):
            # Different metrics.collections classes accept different subsets
            # of these fields (e.g. ResponseGroundedness only takes
            # `response` + `retrieved_contexts`, ContextRelevance only takes
            # `user_input` + `retrieved_contexts`). Passing a field a given
            # metric doesn't declare raises "unexpected keyword argument",
            # so filter against the real signature instead of guessing.
            candidate_kwargs = {
                "user_input": sample.user_input,
                "response": sample.response,
                "reference": sample.reference,
                "retrieved_contexts": sample.retrieved_contexts,
            }
            accepted_params = inspect.signature(scorer.ascore).parameters
            kwargs = {
                k: v
                for k, v in candidate_kwargs.items()
                if k in accepted_params and v
            }

            res = await scorer.ascore(**kwargs)
            if hasattr(res, "value"):
                return (float(res.value) if res.value is not None else 0.0), None
            return (float(res) if res is not None else 0.0), None

        # 3. Fallback to synchronous score() method if available
        elif hasattr(scorer, "score"):
            res = scorer.score(
                user_input=sample.user_input,
                response=sample.response,
                reference=sample.reference,
            )
            if hasattr(res, "value"):
                return (float(res.value) if res.value is not None else 0.0), None
            return (float(res) if res is not None else 0.0), None

        else:
            return 0.0, f"No usable scoring method found on {metric_name}"

    except Exception as e:
        # Print the FULL traceback (not just str(e)) to the console running
        # Streamlit, and also surface a short error message back to the
        # caller so it shows up in the results table instead of a
        # misleadingly clean-looking 0.0.
        print(f"--- Error scoring {metric_name} ---")
        traceback.print_exc()
        return 0.0, f"{type(e).__name__}: {e}"


def run_comprehensive_evaluations(model_id: str) -> pd.DataFrame:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from environment.")

    # 1. Initialize Async OpenAI client pointing to Groq API
    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )

    # 2a. LLM for the "modern" ragas.metrics.collections classes
    #     (AnswerAccuracy, ContextRelevance, ResponseGroundedness). These
    #     are specifically designed to work with llm_factory's client-based
    #     interface.
    llm = llm_factory(model_id, provider="openai", client=client)

    # 2b. LLM for the "legacy" single-turn metrics (AspectCritic,
    #     SimpleCriteriaScore, FactualCorrectness, ResponseRelevancy,
    #     LLMContextRecall). These expect a BaseRagasLLM, not the
    #     llm_factory client wrapper above — passing the wrong type here is
    #     what was silently throwing on every one of these metrics.
    legacy_llm = LangchainLLMWrapper(
        ChatGroq(model=model_id, api_key=api_key)
    )

    # 3. Define Metric Suite
    benchmark_suite = [
        # --- GENERAL PURPOSE ---
        {
            "category": "General Purpose",
            "metric_name": "Aspect Critic (Conciseness)",
            "scorer": AspectCritic(
                name="conciseness",
                definition="Is the response concise and free from unnecessary filler?",
                llm=legacy_llm,
            ),
            "user_input": "What is gravity?",
            "reference": "Gravity is a fundamental force pulling objects toward each other.",
            "contexts": [
                "Gravity is a fundamental force by which a planet or other body draws objects toward its center."
            ],
        },
        {
            "category": "General Purpose",
            "metric_name": "Simple Criteria Score",
            "scorer": SimpleCriteriaScore(
                name="clarity_score",
                definition="Score the clarity and readability of the response on a scale of 0 to 1.",
                llm=legacy_llm,
            ),
            "user_input": "Explain quantum physics simply.",
            "reference": "Quantum physics studies tiny particles like atoms and photons.",
            "contexts": [
                "Quantum mechanics is a fundamental theory in physics that describes nature at the scale of atomic systems."
            ],
        },
        # --- RAG METRICS ---
        {
            "category": "RAG Metrics",
            "metric_name": "Factual Correctness",
            "scorer": FactualCorrectness(llm=legacy_llm),
            "user_input": "When was Einstein born?",
            "reference": "Albert Einstein was born in 1879.",
            "contexts": ["Albert Einstein was born on 14 March 1879."],
        },
        {
            "category": "RAG Metrics",
            "metric_name": "Response Relevancy",
            "scorer": ResponseRelevancy(
                llm=legacy_llm, embeddings=embeddings, strictness=1
            ),
            "user_input": "What are the benefits of solar energy?",
            "reference": "Solar energy is renewable and reduces electricity costs.",
            "contexts": [
                "Solar power is clean, abundant, and reduces reliance on fossil fuels."
            ],
        },
        {
            "category": "RAG Metrics",
            "metric_name": "Context Recall",
            "scorer": LLMContextRecall(llm=legacy_llm),
            "user_input": "Where was Albert Einstein born?",
            "reference": "Albert Einstein was born in Ulm, Germany.",
            "contexts": [
                "Albert Einstein was born March 14, 1879 at Ulm, in Württemberg, Germany."
            ],
        },
        # --- NVIDIA METRICS ---
        {
            "category": "Nvidia Metrics",
            "metric_name": "Answer Accuracy",
            "scorer": AnswerAccuracy(llm=llm),
            "user_input": "When was Einstein born?",
            "reference": "Albert Einstein was born in 1879.",
            "contexts": ["Albert Einstein was born on 14 March 1879."],
        },
        {
            "category": "Nvidia Metrics",
            "metric_name": "Context Relevance",
            "scorer": ContextRelevance(llm=llm),
            "user_input": "When and where was Albert Einstein born?",
            "reference": "Albert Einstein was born in 1879 in Ulm, Germany.",
            "contexts": [
                "Albert Einstein was born March 14, 1879 at Ulm, in Württemberg, Germany."
            ],
        },
        {
            "category": "Nvidia Metrics",
            "metric_name": "Response Groundedness",
            "scorer": ResponseGroundedness(llm=llm),
            "user_input": "Where was Albert Einstein born?",
            "reference": "Albert Einstein was born in Ulm, Germany.",
            "contexts": [
                "Albert Einstein was born March 14, 1879 at Ulm, in Württemberg, Germany."
            ],
        },
    ]

    records = []

    # Sync wrapper to query model and score sample via asyncio
    async def process_test_case(test):
        user_input = test["user_input"]
        reference = test["reference"]
        contexts = test["contexts"]

        # Step 1: Query Groq model asynchronously
        completion = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": user_input}],
        )
        generated_response = completion.choices[0].message.content

        # Step 2: Build SingleTurnSample
        sample = SingleTurnSample(
            user_input=user_input,
            response=generated_response,
            reference=reference,
            retrieved_contexts=contexts,
        )

        # Step 3: Run async metric evaluation
        score_val, error = await evaluate_sample_async(test["scorer"], sample)

        return {
            "Category": test["category"],
            "Metric": test["metric_name"],
            "User Input": user_input,
            "Generated Response": generated_response,
            "Reference / Context": reference,
            "Score": f"{score_val:.4f}" if error is None else "N/A",
        }

    async def main_eval_loop():
        return await asyncio.gather(
            *[process_test_case(test) for test in benchmark_suite]
        )

    # Run loop synchronously for Streamlit
    records = asyncio.run(main_eval_loop())
    return pd.DataFrame(records)