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
git clone https://github.com/sinh57/NeuroChat.git
cd NeuroChat

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

## ☁️ Deploy to Streamlit Cloud (Recommended)

### Step-by-Step Deployment:

1. **Push to GitHub** (already done at https://github.com/sinh57/NeuroChat.git)

2. **Create Streamlit Cloud App:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **New app**
   - Select repository: `sinh57/NeuroChat`
   - Select branch: `main`
   - Select main file: `app.py`
   - Click **Deploy**

3. **Configure API Key:**
   - After deployment, go to your app settings
   - Navigate to **Advanced** → **Secrets**
   - Add your API key:
     ```toml
     OPENAI_API_KEY = "sk-your-openai-key"
     # OR
     GROQ_API_KEY = "gsk-your-groq-key"
     ```
   - Click **Save**

4. **Verify Deployment:**
   - Wait 2-3 minutes for the app to start
   - Check the deployment logs for errors
   - Test the app by sending a message

**Streamlit Cloud Features:**
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Built-in authentication (optional)
- ✅ Environment variables via Secrets
- ✅ Easy redeployment on git push

---

## 🤗 Deploy to HuggingFace Spaces

### Step-by-Step Deployment:

1. **Create New Space:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Space name: `neurochat` (or your choice)
   - License: MIT
   - SDK: Streamlit
   - Hardware: CPU Basic (free)
   - Click **Create Space**

2. **Clone and Push:**
   ```bash
   # Clone your new space
   git clone https://huggingface.co/spaces/yourusername/neurochat
   cd neurochat

   # Copy your project files
   # (or push from your local directory)
   git remote set-url origin https://huggingface.co/spaces/yourusername/neurochat
   git add .
   git commit -m "Deploy NeuroChat to HuggingFace Spaces"
   git push origin main
   ```

3. **Configure API Key:**
   - Go to your Space → **Settings** → **Repository secrets**
   - Add new secret:
     - Name: `OPENAI_API_KEY` or `GROQ_API_KEY`
     - Value: Your actual API key
   - Click **Add new secret**

4. **Verify Deployment:**
   - The Space will automatically rebuild
   - Check the "Logs" tab for errors
   - Test the app once it's running

**HuggingFace Spaces Features:**
- ✅ Free CPU tier
- ✅ GPU options (paid)
- ✅ Community visibility
- ✅ Repository secrets
- ✅ Automatic rebuilds

---

## 🔧 Environment Configuration

### Required Environment Variables:

| Variable | Description | Required |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes (if using OpenAI) |
| `GROQ_API_KEY` | Groq API key for Llama models | Yes (if using Groq) |

### Local Development (.env file):
```env
# .env file (never commit this)
OPENAI_API_KEY=sk-your-openai-key
# OR
GROQ_API_KEY=gsk-your-groq-key
```

### Deployment Secrets:

**Streamlit Cloud:**
- Settings → Advanced → Secrets
- Add as TOML format: `OPENAI_API_KEY = "your-key"`

**HuggingFace Spaces:**
- Settings → Repository secrets
- Add as key-value pairs

---

## 🐛 Troubleshooting

### Common Issues:

**Issue:** "ModuleNotFoundError: No module named 'langchain.agents'"
- **Solution:** Dependencies are pinned to LangChain 0.2.x for compatibility. Reinstall with `pip install -r requirements.txt`

**Issue:** "Failed to call a function" with Groq models
- **Solution:** Groq models use ReAct agent pattern. If issues persist, switch to OpenAI models (gpt-4o-mini)

**Issue:** App fails to start on deployment
- **Solution:** Check deployment logs. Ensure all dependencies are in requirements.txt

**Issue:** API key not working
- **Solution:** Verify API key format:
  - OpenAI: starts with `sk-`
  - Groq: starts with `gsk-` or `gsk_`

**Issue:** SQLite database errors on deployment
- **Solution:** The database is created automatically. Ensure the app has write permissions

**Issue:** Knowledge base not working
- **Solution:** Ensure FAISS and sentence-transformers are installed. Check if documents are uploaded correctly

### Deployment Checklist:

- [ ] API key configured in secrets
- [ ] requirements.txt includes all dependencies
- [ ] .env is in .gitignore
- [ ] conversations.db is in .gitignore
- [ ] knowledge_base/ is in .gitignore
- [ ] README.md has correct repository URL
- [ ] App runs locally without errors
- [ ] Test with both OpenAI and Groq models

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
