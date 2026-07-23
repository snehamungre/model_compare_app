# frontend/eval_view.py
from backend.evaluation import run_nvidia_evaluations
import streamlit as st


def render_eval_tab(selected_model_id: str):
    st.header("Nvidia Ragas Metrics Evaluation")
    st.caption(
        f"Evaluating predefined benchmark samples using active model: `{selected_model_id}`"
    )

    st.write("""
    This tab runs standard Nvidia evaluation benchmarks (**Answer Accuracy**, **Context Relevance**, and **Response Groundedness**) 
    by querying the selected Groq model for predetermined inputs and scoring the generated responses.
    """)

    if st.button("🚀 Run Nvidia Metrics Evaluation"):
        with st.spinner(
            f"Generating responses and calculating Nvidia metrics for `{selected_model_id}`..."
        ):
            df_results = run_nvidia_evaluations(model_id=selected_model_id)

            if df_results.empty:
                st.error("Evaluation returned no results.")
            else:
                st.success("Evaluation complete!")
                st.subheader("📊 Evaluation Benchmark Summary")
                # Render DataFrame without index column
                st.dataframe(df_results, hide_index=True, width="stretch")
