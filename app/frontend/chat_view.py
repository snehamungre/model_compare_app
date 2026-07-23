from backend.model_analytics import calculate_cost_and_metrics
import pandas as pd
import streamlit as st


def render_chat_tab(client, option, system_prompt, model_dict):
    """Renders Chat interface and Session Analytics Table."""

    # Initialize Session State Variables
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "analytics_history" not in st.session_state:
        st.session_state.analytics_history = []

    def response_generator(prompt_text):
        messages = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        completion = client.chat.completions.create(
            messages=messages,
            model=option,
            stream=True,
        )

        for chunk in completion:
            if hasattr(chunk, "usage") and chunk.usage:
                st.session_state["latest_usage"] = chunk.usage

            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Assistant Output
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt), cursor="▌")
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Single Message Analysis
        usage = st.session_state.get("latest_usage")
        analysis_text, record = calculate_cost_and_metrics(usage, option, model_dict)

        with st.chat_message("system", avatar="📊"):
            st.write(analysis_text)

        st.session_state.messages.append({"role": "system", "content": analysis_text})

        if record:
            st.session_state.analytics_history.append(record)

    # Session Summary Table
    st.markdown("---")
    st.subheader("📊 Session Analytics Summary")

    if st.session_state.analytics_history:
        df = pd.DataFrame(st.session_state.analytics_history)
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.info("No queries sent yet. Send a message above to generate data.")
