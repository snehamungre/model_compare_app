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

## 📁 Project Structure

```text
AI_model_compare_app/
├── app/
│   ├── main.py          # Streamlit UI & chat loop
│   └── models.py       # Groq API model fetching & analytics calculation
├── .env                 # API Key configuration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
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