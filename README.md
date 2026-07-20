# 🤖 Groq AI Model Comparator & Chatbot

An interactive Streamlit web application that lets you chat with various AI models powered by the Groq API while analyzing real-time metrics—including token consumption, generation speed ($\text{tokens/sec}$), latency, and per-prompt inference costs.

---

## ✨ Features

* **Multi-Model Selector:** Easily switch between all available Groq models (Llama, GPT-OSS, Qwen, Orpheus, etc.).
* **Dynamic API & Cache Management:** Automatically fetches available models and pricing metadata from Groq with local caching (`st.cache_data`) to prevent redundant API calls.
* **Real-Time Output Streaming:** Responses stream smoothly using `st.write_stream` and Groq's native streaming endpoint.
* **In-Depth Cost & Usage Analysis:**
  * Prompt vs. completion token breakdown.
  * Real-time generation speed ($\text{tokens/sec}$).
  * Total latency ($\text{seconds}$).
  * Exact cost calculation per request based on live token rates.
  * Cumulative session spending tracker.

---

## 🛠️ Project Structure

```text
AI_model_compare_app/
├── app/
│   ├── main.py          # Streamlit UI & chat loop
│   └── chatbot.py       # Groq API model fetching & caching logic
├── .env                 # Environment variables (API Key)
├── .gitignore           # Git ignore file
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or higher installed.
* A **Groq API key** (obtainable from [Groq Console](https://console.groq.com/)).

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/AI_model_compare_app.git](https://github.com/your-username/AI_model_compare_app.git)
cd AI_model_compare_app
```

### 2. Set Up Virtual Environment

```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the Application

```bash
streamlit run app/main.py
```

Open your browser to `http://localhost:8501` to view the app.

---

## 📊 Analytics Calculated

* **Token Breakdown:** `Total Tokens = Prompt Tokens + Completion Tokens`
* **Generation Speed:** $\text{TPS} = \frac{\text{Completion Tokens}}{\text{Completion Time (seconds)}}$
* **Request Cost:** $\text{Cost} = (\text{Prompt Tokens} \times \text{Prompt Rate}) + (\text{Completion Tokens} \times \text{Completion Rate})$