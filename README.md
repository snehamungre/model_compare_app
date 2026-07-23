# 🤖 Groq AI Model Comparator & Chatbot

An interactive Streamlit application that allows you to chat with various AI models powered by the Groq API while tracking real-time performance metrics—including token consumption, latency, and estimated request costs. Every prompt's analytics are automatically aggregated into a clean session summary table.

---

## ✨ Features

* **Multi-Model Support:** Easily switch between all available Groq models (Llama 3, GPT-OSS, Qwen, etc.) via the sidebar.
* **System Prompt Customization:** Test system prompt adherence directly within the chat interface.
* **Real-Time Streaming:** Smooth response generation powered by Groq's low-latency inference engine and Streamlit's `write_stream`.
* **Automatic Price & Metric Calculations:**
  * Displays prompt tokens, completion tokens, and total usage per message.
  * Measures execution latency (seconds).
  * Calculates exact USD costs dynamically based on model pricing metadata.
* **Session Analytics Summary Table:** Automatically appends metrics for every interaction into an aggregated, index-free Pandas DataFrame displayed underneath the chat.
* **API Response Caching:** Uses `@lru_cache` to fetch Groq model lists and pricing without redundant network calls.
---
## 🧪 Ragas Model Evaluation & Benchmarking

The application includes an automated evaluation pipeline powered by **[Ragas](https://docs.ragas.io/)** that benchmarks model performance across multiple standardized evaluation frameworks using Groq as an LLM judge.

### 📌 Supported Metric Suites

| Category | Metric | Description |
| :--- | :--- | :--- |
| **General Purpose** | **Aspect Critic (Conciseness)** | Assesses if the response is direct, concise, and free of unnecessary filler text. |
| | **Simple Criteria Score** | Evaluates overall clarity and readability against natural language criteria. |
| **RAG Quality** | **Factual Correctness** | Measures factual agreement between generated responses and ground truth references. |
| | **Response Relevancy** | Quantifies how directly the answer addresses the user's initial query. |
| | **Context Recall** | Checks whether all relevant ground-truth facts were successfully retrieved. |
| **Nvidia Metrics** | **Answer Accuracy** | Scores exact correctness against reference ground truth. |
| | **Context Relevance** | Evaluates if retrieved context passages directly answer the prompt. |
| | **Response Groundedness** | Ensures answer claims are strictly grounded within provided reference context. |

---

### 🚀 How It Works

1. **Model Selection**: Choose any Groq-hosted model (e.g., `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`) from the sidebar.
2. **Automated Generation & Scoring**: 
   - Queries the selected model asynchronously for standard benchmark questions.
   - Evaluates the generated output against predefined ground truth and context using `SingleTurnSample` wrappers.
   - Leverages `AsyncOpenAI` for concurrent evaluation across all metrics.
3. **Structured Results**: Displays scores in an interactive summary table directly inside the Streamlit dashboard.

---

## 📁 Project Structure

```text
AI_model_compare_app/
├── backend/
│   ├── analytics.py        # Token usage, latency & cost calculations
│   ├── evaluation.py       # Async Ragas & Nvidia evaluation metrics pipeline
│   └── groq_client.py      # Groq API client initialization & model loader
│
├── frontend/
│   ├── chat_view.py        # Chatbot interface & session analytics summary table
│   └── eval_view.py        # Benchmark evaluation UI tab
│
├── main.py                 # App entrypoint, page config & layout tabs
├── .env                    # API keys configuration
├── requirements.txt        # Dependencies (ragas, groq, openai, streamlit, pandas)
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or higher.
* A **Groq API key** (available from the [Groq Console](https://console.groq.com/)).

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

Open your browser at `http://localhost:8501`.

---

## 📊 Session Analytics Tracking

Each request records the following attributes into the summary DataFrame:

| Column | Description |
| :--- | :--- |
| **Model** | Selected Groq model ID |
| **Prompt Tokens** | Input tokens consumed |
| **Completion Tokens** | Output tokens generated |
| **Total Tokens** | Total request token count |
| **Latency (s)** | Time to complete generation |
| **Cost ($)** | Estimated cost based on per-token pricing |