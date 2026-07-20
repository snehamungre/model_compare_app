import os
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import streamlit as st
from models import calculate_cost_and_metrics, get_models

# Load API Key
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Missing GROQ_API_KEY. Set it in your environment or .env file.")

client = Groq(api_key=api_key)

# Set Title
st.title("Model Comparision chatbot")

# Fetch available models
model_ids, model_dict = get_models(api_key)

# Sidebar System Controls
st.sidebar.header("System Configuration:")
option = st.sidebar.selectbox("Which model do you like best?", model_ids)

system_prompt = st.sidebar.text_area(
    "System Prompt (Tests Adherence):",
    value="You are a helpful and concise assistant. Answer in 3-4 sentences.",
    height=150,
)


# Streaming Response Generator
def response_generator(prompt_text):
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
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


# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "analytics_history" not in st.session_state:
    st.session_state.analytics_history = []

# Display chat message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question..."):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Stream Assistant Response
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt), cursor="▌")
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 3. Calculate and display single-message analysis
    usage = st.session_state.get("latest_usage")
    analysis_text, record = calculate_cost_and_metrics(usage, option, model_dict)

    with st.chat_message("system", avatar="📊"):
        st.write(analysis_text)

    st.session_state.messages.append({"role": "system", "content": analysis_text})

    # 4. Append record to history table
    if record:
        st.session_state.analytics_history.append(record)


st.markdown("---")
st.subheader("📊 Session Analytics Summary")

if st.session_state.analytics_history:
    df = pd.DataFrame(st.session_state.analytics_history)

    # Render dataframe with no index column
    st.dataframe(df, hide_index=True, use_container_width=True)
else:
    st.info("No queries sent yet. Send a message to generate data.")
