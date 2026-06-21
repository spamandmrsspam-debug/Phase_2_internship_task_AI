# 🤖 Task 4 — Context-Aware Chatbot Using LangChain + RAG

A conversational chatbot that remembers context across turns and retrieves answers
from a vectorized knowledge base, built with LangChain, FAISS, HuggingFace Inference API,
and Streamlit.

---

## ✅ Task Requirements Coverage

| Requirement | Status |
|---|---|
| Use LangChain or RAG | ✅ Full RAG pipeline — FAISS retrieval + Mistral-7B generation |
| Context memory for conversational history | ✅ Last 6 turns injected into every prompt via `st.session_state` |
| Retrieve answers from vectorized document store | ✅ FAISS vector store, top-3 chunks retrieved per query |
| Deploy with Streamlit | ✅ Full chat UI at `localhost:8501` with source viewer and sidebar |

---

## 📁 Project Structure

```
task4_chatbot/
├── chatbot_app.py     # Main Streamlit app — run this
├── requirements.txt   # Dependencies
├── .env.example       # Token template — copy to .env
└── README.md          # This file
```

---

## 🏗️ Architecture

```
User Query
    │
    ▼
st.session_state.chat_history  ←── Context Memory (last 6 turns)
    │
    ▼
FAISS Vector Store ──── HuggingFaceEmbeddings (all-MiniLM-L6-v2, local/free)
    │   top-3 chunks retrieved via cosine similarity
    ▼
RAG Prompt (history + retrieved context + question)
    │
    ▼
HuggingFace InferenceClient → Mistral-7B-Instruct-v0.2 (free API)
    │
    ▼
Answer + Retrieved Source Chunks displayed in Streamlit UI
```

---

## 📚 Knowledge Base (Custom Corpus)

The chatbot is pre-loaded with a built-in AI/ML corpus covering 10 topics:

- Machine Learning fundamentals (supervised, unsupervised, reinforcement)
- Deep Learning & Neural Networks (CNN, RNN, Transformer, GAN)
- NLP & Large Language Models (BERT, GPT)
- Retrieval-Augmented Generation (RAG pipeline, vector DBs)
- Scikit-learn Pipelines (ColumnTransformer, GridSearchCV)
- Model Evaluation Metrics (F1, ROC-AUC, MAE, RMSE)
- Transfer Learning & Fine-tuning (LoRA, PEFT)
- Customer Churn Prediction (Telco dataset, joblib)
- Embeddings & FAISS (similarity search, vector databases)

> To use your own documents, replace the `CORPUS` list in `chatbot_app.py`
> with content from your PDFs, Wikipedia pages, or internal documents.

---

## ⚙️ Setup

### Step 1 — Get a Free HuggingFace Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → name it (e.g. "RAG Chatbot")
3. Under **Inference**, tick ✅ **"Make calls to Inference Providers"**
4. Save and copy the token (starts with `hf_`)

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

First run downloads the embedding model (~80MB). This only happens once.

### Step 4 — Add Your Token (Optional)

```bash
# Copy the example file and fill in your token
cp .env.example .env
```

You can also paste the token directly in the Streamlit sidebar — no `.env` file needed.

---

## ▶️ How to Run

```bash
streamlit run chatbot_app.py
```

Opens at **http://localhost:8501** automatically.

> ⚠️ Always use `streamlit run` — not `python chatbot_app.py`.
> Running with Python directly causes `missing ScriptRunContext` warnings and the app won't work.

**First run:** paste your HuggingFace token in the sidebar. The knowledge base
builds in ~30 seconds (cached — instant on all subsequent runs in the same session).

---

## 💬 Example Questions to Try

```
What is AI?
What is the difference between supervised and unsupervised learning?
How does BERT work?
What is RAG and why is it useful?
What metrics should I use for an imbalanced dataset?
How does FAISS perform similarity search?
What is transfer learning?
Tell me more about that          ← tests context memory (follow-up question)
```

---

## 🔑 API Key Info

| Service | Cost | Required |
|---|---|---|
| HuggingFace Inference API (Mistral-7B) | **Free** (rate limited) | ✅ Yes (free account) |
| HuggingFace Embeddings (all-MiniLM-L6-v2) | **Free** (runs locally) | ❌ No key needed |
| OpenAI | Paid | ❌ Not used |

> If you see HTTP 429, HuggingFace's free rate limit was hit — wait ~1 minute and retry.

---

## 🧠 Skills Gained

- Retrieval-Augmented Generation (RAG) pipeline construction
- Document embedding and vector search with FAISS
- LangChain ecosystem (text splitters, embeddings, vector stores)
- HuggingFace `InferenceClient` for LLM integration
- Context-aware multi-turn conversation via prompt engineering
- Streamlit chat UI (`st.chat_message`, `st.session_state`, `st.cache_resource`)
- Secret management with `python-dotenv`

---

## 📌 References

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/qa_chat_history/)
- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [HuggingFace InferenceClient docs](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client)
- [Streamlit Chat Elements](https://docs.streamlit.io/develop/api-reference/chat)
