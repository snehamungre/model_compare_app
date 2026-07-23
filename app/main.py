# main.py
import os
from backend.groq_client import get_models
from dotenv import load_dotenv
from frontend.chat_view import render_chat_tab
from frontend.evaluation_view import render_eval_tab
from groq import Groq
import streamlit as st

# Load API Key
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Missing GROQ_API_KEY. Set it in your environment or .env file.")

client = Groq(api_key=api_key)

st.set_page_config(page_title="Groq AI Lab & Nvidia Ragas Evaluation", layout="wide")
st.title("🤖 Groq AI Lab")

# Fetch available models
model_ids, model_dict = get_models(api_key)

# Sidebar Model Configuration
st.sidebar.header("⚙️ Configuration")
option = st.sidebar.selectbox("Select Model:", model_ids)
system_prompt = st.sidebar.text_area(
    "System Prompt (Tests Adherence):",
    value="You are a helpful and concise assistant.",
    height=100,
)

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analytics_history" not in st.session_state:
    st.session_state.analytics_history = []

# Navigation Tabs
tab1, tab2 = st.tabs(["Chat & Analytics Table", "Metrics Evaluation"])

with tab1:
    render_chat_tab(client, option, system_prompt, model_dict)

with tab2:
    render_eval_tab(selected_model_id=option)
