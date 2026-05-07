---
title: NeuroChat AI Agent
emoji: 🧠
colorFrom: purple
colorTo: cyan
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
python_version: 3.11
---

# 🧠 NeuroChat — Conversational AI with Memory & Tools

> A production-ready AI agent built with **LangChain**, **LangGraph**, and **Streamlit**.  
> Features persistent conversation memory, RAG knowledge base, 6 integrated tools, and a dark-themed chat UI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **Persistent Memory** | Buffer, Summary, and Window memory types — switchable at runtime |
| 💾 **SQLite Persistence** | Save/load conversations across sessions with custom names |
| 📖 **RAG Knowledge Base** | Upload PDF/DOCX/TXT documents for AI to query via vector search |
| 🔧 **6 Tools** | Web search, calculator, Wikipedia, datetime, live weather, knowledge base |
| 🗺️ **LangGraph Workflow** | Typed `AgentState`, single compiled graph, memory outside graph |
| 💬 **Streamlit UI** | Dark-themed chat, real-time tool badges, session stats |
| ⚙️ **Runtime Config** | Switch model, temperature, tools, memory — no restart needed |
| 🤖 **Dual API Support** | OpenAI and Groq (Llama) models with auto-detection |
| 🚀 **Deploy-ready** | Streamlit Cloud + HuggingFace Spaces config included |

---

## 🏗️ Architecture

```
neurochat/
├── app.py                  # Streamlit UI + session management
├── agent/
│   ├── graph.py            # LangGraph StateGraph + agent node factory
│   └── tools.py            # 6 tool implementations (including RAG)
├── utils/
│   ├── db.py               # SQLite persistence for conversations
│   ├── rag.py              # RAG pipeline with FAISS + sentence-transformers
│   └── helpers.py          # Shared utilities
├── .streamlit/
│   └── config.toml         # Theme + server config (deploy-ready)
├── requirements.txt
├── .env.example
└── README.md
```

### LangGraph Flow

```
          ┌─────────────────────────────────────────┐
          │            AgentState (TypedDict)        │
          │  input · chat_history · output ·         │
          │  tools_used                              │
          └──────────────┬──────────────────────────┘
                         │
                         ▼
                   [ agent_node ]
                         │
              ┌──────────┴──────────┐
              │   AgentExecutor     │
              │  (OpenAI/ReAct)     │
              │                     │
              │  Tool calls?        │
              │  ┌───────────────┐  │
              │  │ calculator    │  │
              │  │ web_search    │  │
              │  │ wikipedia     │  │
              │  │ datetime      │  │
              │  │ weather       │  │
              │  │ knowledge_base│  │
              │  └───────────────┘  │
              └──────────┬──────────┘
                         │
              Memory.save_context()
              SQLite.save_message()
                         │
                        END
```

### Why memory lives outside the graph

LangGraph recompiles the graph on `invoke()` — if memory were created inside, it would reset every turn. By storing the memory object in `st.session_state` and passing it into the node via closure, it survives all Streamlit reruns.

---

## 🚀 Local Setup

```bash
# 1. Clone
git clone https://github.com/yourusername/neurochat
cd neurochat

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY or GROQ_API_KEY

# 5. Run
streamlit run app.py
```

**Supported Models:**
- **OpenAI:** gpt-4o-mini, gpt-4o, gpt-3.5-turbo
- **Groq:** llama-3.3-70b-versatile, llama-3.1-70b-versatile, gemma2-9b-it

---

## ☁️ Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo → `app.py` as entry point
4. Under **Advanced settings → Secrets**, add your API key:
   ```toml
   OPENAI_API_KEY = "sk-your-key"
   # OR
   GROQ_API_KEY = "gsk-your-key"
   ```
5. Click **Deploy** — live in ~2 minutes ✅

---

## 🤗 Deploy to HuggingFace Spaces (free)

The `---` header at the top of this README is the HF Spaces config.

```bash
# Create a new Space at huggingface.co/new-space
# SDK: Streamlit
git remote add hf https://huggingface.co/spaces/yourusername/neurochat
git push hf main
```
Add `OPENAI_API_KEY` or `GROQ_API_KEY` in Space → Settings → Repository secrets.

---

## 🧬 Memory Types Explained

| Type | How it works | Best for |
|---|---|---|
| **ConversationBuffer** | Stores every message verbatim | Short–medium chats |
| **ConversationSummary** | LLM summarises old turns | Long conversations |
| **ConversationWindow** | Keeps last *k* turns only | Focused, token-efficient |

---

## 🔧 Tools

| Tool | Source | API key? |
|---|---|---|
| 🌐 Web Search | DuckDuckGo | ❌ Free |
| 🔢 Calculator | Built-in (safe eval) | ❌ |
| 📚 Wikipedia | `wikipedia` library | ❌ Free |
| 🕐 DateTime | `datetime` stdlib | ❌ |
| 🌤️ Weather | wttr.in | ❌ Free |
| 📖 Knowledge Base | FAISS + sentence-transformers | ❌ Local |

---

## 💾 Conversation Persistence

Conversations are automatically saved to SQLite when enabled:
- Toggle "Save this conversation" in the sidebar
- Name your conversation with custom titles
- Load previous conversations anytime
- Conversations persist across app restarts

---

## 📖 RAG Knowledge Base

Upload documents (PDF, DOCX, TXT) to create a searchable knowledge base:
- Documents are chunked and embedded using sentence-transformers
- Stored in FAISS vector store for fast similarity search
- Enable "Knowledge Base" tool to let the AI query your documents
- Clear knowledge base anytime to reset

---

## 🤖 Model Support

**OpenAI Models:**
- gpt-4o-mini (recommended for cost + performance)
- gpt-4o (best quality)
- gpt-3.5-turbo (fastest)

**Groq Models (Llama):**
- llama-3.3-70b-versatile (best quality)
- llama-3.1-70b-versatile
- gemma2-9b-it (fastest)

**Note:** Groq models use ReAct agent pattern for better compatibility with Llama. OpenAI models use native function calling.

---

## 📄 Resume Bullets

```
• Built a production-ready conversational AI agent using LangChain and LangGraph
  with a stateful StateGraph, typed AgentState schema, and persistent cross-turn
  memory (Buffer / Summary / Window).

• Implemented SQLite-based conversation persistence with custom naming, allowing
  users to save, load, and manage conversations across sessions.

• Built a RAG pipeline using FAISS and sentence-transformers, enabling users to
  upload PDF/DOCX/TXT documents for AI to query via vector similarity search.

• Integrated 6 tools (DuckDuckGo search, calculator, Wikipedia, datetime, weather,
  knowledge base) with dual agent patterns: OpenAI function-calling and ReAct for
  Groq/Llama models.

• Added dual API support for OpenAI and Groq (Llama) models with auto-detection,
  providing users with cost-effective and high-performance options.

• Architected memory lifecycle correctly — stored LangChain memory outside the
  LangGraph compile cycle to prevent per-turn reset, a common production pitfall.

• Shipped a Streamlit chat UI with real-time tool-call badges, session stats,
  runtime model/memory switching, and knowledge base management; deployed to
  Streamlit Cloud and HuggingFace Spaces.
```

---

## 🔮 Roadmap

- [x] RAG pipeline with FAISS vector store
- [x] Persistent memory across sessions (SQLite)
- [ ] Multi-agent graph (Planner → Researcher → Writer)
- [ ] Voice input/output (Whisper + TTS)
- [ ] Streaming token output
- [ ] File upload/processing (PDF, images)
- [ ] User authentication
- [ ] Conversation export (JSON, Markdown)
- [ ] Analytics dashboard

---

## 📜 License

MIT — free to use, fork, and modify.
