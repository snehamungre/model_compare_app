from statistics import mode
import time
import os
from groq import Groq
import streamlit as st
from dotenv import load_dotenv
from chatbot import calculate_cost_and_metrics, get_models

# get the API key and access the client
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Missing GROQ_API_KEY. Set it in your environment or .env file.")

client = Groq(api_key=api_key)

st.title("Model Compare Bot")

# get the models avalible
model_ids, model_dict = get_models(api_key)
option = st.selectbox("Which number do you like best?", model_ids)


# Streamed response emulator
def response_generator(prompt):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=option,
    )
    for word in response.choices[0].message.content.split():
        yield word + " "
        time.sleep(0.08)

    st.session_state["latest_usage"] = response.usage

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt), cursor="▌")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display prompt analysis in chat message container
    usage = st.session_state.get("latest_usage")
    with st.chat_message("system", avatar="⚙️", width="stretch"):
        analysis = calculate_cost_and_metrics(usage, option, model_dict)
        st.write(analysis)

    st.session_state.messages.append({"role": "system", "content": analysis})
