# frontend/eval_view.py
from backend.evaluation import run_comprehensive_evaluations
import streamlit as st


def render_eval_tab(selected_model_id: str):
    st.header("Comprehensive Ragas Evaluation Benchmark")
    st.caption(
        f"Evaluating active model: `{selected_model_id}` across General Purpose, RAG, and Nvidia Metric Suites."
    )

    if st.button("🚀 Run Full Evaluation Benchmark"):
        with st.spinner(
            f"Generating responses and calculating Ragas metrics for `{selected_model_id}`..."
        ):
            df_results = run_comprehensive_evaluations(model_id=selected_model_id)

            if df_results.empty:
                st.error("Evaluation returned no results.")
            else:
                st.success("Evaluation complete!")
                st.subheader("📊 Metric Scores Summary")

                # Render DataFrame formatted without index
                st.dataframe(df_results, hide_index=True, width="stretch")
