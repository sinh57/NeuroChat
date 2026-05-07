"""
app.py — NeuroChat: Conversational AI with Memory & Tools
Stack : LangChain + LangGraph + Streamlit
"""

from typing import Dict, List

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from agent.graph import build_agent
from utils.helpers import memory_label, sanitise
from utils.db import init_db, create_conversation, save_message, get_conversations, get_conversation_messages, delete_conversation, create_conversation_with_title, update_conversation_title
from utils.rag import load_document, split_documents, add_documents_to_store, clear_knowledge_base

load_dotenv()

# Initialize database
init_db()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroChat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');
:root {
    --bg: #0a0a0f; --surface: #111118; --border: #1e1e2e;
    --accent: #7c6af7; --accent2: #3ecfcf; --text: #e2e2f0; --muted: #6b6b8a;
}
html,body,[class*="css"] { background:var(--bg)!important; color:var(--text)!important; font-family:'Sora',sans-serif!important; }
section[data-testid="stSidebar"] { background:var(--surface)!important; border-right:1px solid var(--border)!important; }
.user-msg {
    background:#1a1a2e; border:1px solid var(--accent);
    border-radius:12px 12px 4px 12px; padding:12px 16px; margin:6px 0;
    line-height:1.65; box-shadow:0 0 14px rgba(124,106,247,.15);
}
.ai-msg {
    background:#0f1923; border:1px solid var(--border);
    border-left:3px solid var(--accent2);
    border-radius:4px 12px 12px 12px; padding:12px 16px; margin:6px 0; line-height:1.65;
}
.tool-badge {
    display:inline-block; background:rgba(62,207,207,.1);
    border:1px solid var(--accent2); color:var(--accent2);
    font-family:'Space Mono',monospace; font-size:.7rem;
    padding:2px 9px; border-radius:20px; margin:2px 2px 0 0;
}
.mem-chip {
    background:rgba(124,106,247,.1); border:1px solid var(--accent);
    color:var(--accent); font-size:.72rem; padding:3px 10px;
    border-radius:20px; font-family:'Space Mono',monospace;
}
h1 { font-family:'Space Mono',monospace!important; color:var(--accent)!important; letter-spacing:-1px; }
.stButton>button {
    background:var(--accent)!important; color:#fff!important; border:none!important;
    border-radius:8px!important; font-family:'Sora',sans-serif!important;
    font-weight:600!important; transition:all .2s!important;
}
.stButton>button:hover { background:#9b8df9!important; transform:translateY(-1px)!important; box-shadow:0 4px 18px rgba(124,106,247,.45)!important; }
.stTextInput>div>div>input {
    background:var(--surface)!important; border:1px solid var(--border)!important;
    color:var(--text)!important; border-radius:8px!important;
}
.stTextInput>div>div>input:focus { border-color:var(--accent)!important; box-shadow:0 0 0 2px rgba(124,106,247,.25)!important; }
div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:10px; }
</style>
""", unsafe_allow_html=True)


# ── Resolve API key (sidebar input OR Streamlit secrets) ──────────────────────
def _get_secret_key() -> str:
    """Read API key from st.secrets if available (Streamlit Cloud) or environment (HF Spaces)."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    
    return os.environ.get("GROQ_API_KEY", os.environ.get("OPENAI_API_KEY", ""))


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: Dict = {
        "messages":     [],   # [{role, content, tools_used}]
        "chat_history": [],   # serialised [{role, content}]
        "graph":        None,
        "memory":       None,
        "tool_log":     [],
        "cfg":          {},
        "conversation_id": None,  # Current conversation ID for persistence
        "save_conversation": False,  # Whether to save current conversation
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧠 NeuroChat")
    st.caption("Conversational AI · Memory · Tools")
    st.divider()

    st.markdown("### ⚙️ Configuration")

    # API key — only show input if not already configured via secrets
    secret_key = _get_secret_key()
    if secret_key:
        st.info("✅ API key configured (hidden)")
        api_key = secret_key
    else:
        api_key = st.text_input(
            "API Key (OpenAI or Groq)",
            type="password",
            placeholder="sk-… or gsk-…",
            help="Your key is never stored. On Streamlit Cloud add it in App Secrets.",
        )

    model = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "gemma2-9b-it", "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

    st.divider()
    st.markdown("### 🛠️ Tools")
    t_search = st.checkbox("🌐 Web Search",  value=True)
    t_calc   = st.checkbox("🔢 Calculator",  value=True)
    t_wiki   = st.checkbox("📚 Wikipedia",   value=True)
    t_time   = st.checkbox("🕐 DateTime",    value=True)
    t_wx     = st.checkbox("🌤️ Weather",     value=True)
    t_kb     = st.checkbox("📖 Knowledge Base", value=False)

    st.divider()
    st.markdown("### 🧬 Memory")
    mem_type = st.selectbox(
        "Memory Type",
        ["ConversationBuffer", "ConversationSummary", "ConversationWindow"],
    )
    window_k = st.slider("Window (k)", 2, 20, 5) if mem_type == "ConversationWindow" else 5

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.tool_log = []
            st.session_state.graph = None
            st.session_state.memory = None
            st.session_state.cfg = {}
            st.session_state.conversation_id = None
            st.session_state.save_conversation = False
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.tool_log = []
            st.rerun()

    st.divider()
    st.markdown("### 💾 Saved Conversations")
    
    # Toggle for saving current conversation
    st.session_state.save_conversation = st.checkbox(
        "Save this conversation",
        value=st.session_state.save_conversation
    )
    
    # Custom conversation name input
    if st.session_state.save_conversation:
        conv_name = st.text_input(
            "Conversation name",
            value=f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            key="conversation_name_input"
        )
        
        # If saving and no conversation ID exists, create one with custom name
        if st.session_state.conversation_id is None:
            if st.session_state.messages:
                st.session_state.conversation_id = create_conversation_with_title(conv_name)
        else:
            # Update title if it changed
            if conv_name:
                update_conversation_title(st.session_state.conversation_id, conv_name)
    
    # Show saved conversations
    conversations = get_conversations()
    if conversations:
        for conv in conversations:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(conv["title"], key=f"load_{conv['id']}", use_container_width=True):
                    # Load conversation
                    messages = get_conversation_messages(conv["id"])
                    st.session_state.messages = messages
                    st.session_state.chat_history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in messages
                    ]
                    st.session_state.conversation_id = conv["id"]
                    st.session_state.save_conversation = True
                    st.session_state.tool_log = []
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{conv['id']}"):
                    delete_conversation(conv["id"])
                    if st.session_state.conversation_id == conv["id"]:
                        st.session_state.conversation_id = None
                        st.session_state.save_conversation = False
                    st.rerun()
    else:
        st.caption("No saved conversations yet")

    st.divider()
    st.markdown("### � Knowledge Base")
    
    # Document upload section
    uploaded_file = st.file_uploader(
        "Upload document (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        help="Upload documents to add to the knowledge base for RAG queries"
    )
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and process document
                documents = load_document(temp_path)
                chunks = split_documents(documents)
                add_documents_to_store(chunks)
                
                # Clean up temp file
                os.remove(temp_path)
                
                st.success(f"✅ Added {len(chunks)} chunks from {uploaded_file.name} to knowledge base")
            except Exception as e:
                st.error(f"❌ Error processing document: {e}")
    
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        clear_knowledge_base()
        st.success("✅ Knowledge base cleared")
        st.rerun()

    st.divider()
    st.markdown("### � Stats")
    c1, c2 = st.columns(2)
    c1.metric("Messages",   len(st.session_state.messages))
    c2.metric("Tool calls", len(st.session_state.tool_log))


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("## 💬 Conversation")

lbl = memory_label(st.session_state.chat_history)
if lbl:
    st.markdown(f'<span class="mem-chip">🧠 {lbl}</span>', unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">👤 <b>You</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        badges = "".join(
            f'<span class="tool-badge">⚡ {t}</span>'
            for t in msg.get("tools_used", [])
        )
        badge_row = f'<div style="margin-top:8px">{badges}</div>' if badges else ""
        st.markdown(
            f'<div class="ai-msg">🧠 <b>NeuroChat</b><br>{msg["content"]}{badge_row}</div>',
            unsafe_allow_html=True,
        )

st.divider()

# Input row
col_in, col_btn = st.columns([5, 1])
with col_in:
    user_input: str = st.text_input(
        "msg", placeholder="Ask me anything…",
        label_visibility="collapsed", key="input_box",
    )
with col_btn:
    send: bool = st.button("Send →", use_container_width=True)

# Quick prompts
st.markdown("**Try:**")
qcols = st.columns(4)
quick: List[str] = [
    "What's 15% of 847?",
    "Explain LangGraph",
    "Weather in Tokyo?",
    "What day is today?",
]
for i, q in enumerate(quick):
    with qcols[i]:
        if st.button(q, key=f"q{i}", use_container_width=True):
            user_input, send = q, True


# ── Handle send ───────────────────────────────────────────────────────────────
if send and user_input.strip():
    if not api_key:
        st.error("⚠️ Enter your OpenAI API key in the sidebar.")
        st.stop()

    user_input = sanitise(user_input)

    active_tools: List[str] = []
    if t_search: active_tools.append("web_search")
    if t_calc:   active_tools.append("calculator")
    if t_wiki:   active_tools.append("wikipedia")
    if t_time:   active_tools.append("datetime")
    if t_wx:     active_tools.append("weather")
    if t_kb:     active_tools.append("knowledge_base")

    # Rebuild agent only when config actually changes
    cfg = {
        "model": model, "temperature": temperature,
        "tools": tuple(active_tools), "mem_type": mem_type, "window_k": window_k,
    }
    if st.session_state.graph is None or st.session_state.cfg != cfg:
        with st.spinner("⚙️ Initialising agent…"):
            graph, memory = build_agent(
                api_key=api_key,
                model=model,
                temperature=temperature,
                selected_tools=active_tools,
                memory_type=mem_type,
                window_k=window_k,
            )
        st.session_state.graph  = graph
        st.session_state.memory = memory
        st.session_state.cfg    = cfg

    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Save user message to database if conversation saving is enabled
    if st.session_state.save_conversation and st.session_state.conversation_id:
        save_message(st.session_state.conversation_id, "user", user_input, [])

    with st.spinner("🧠 Thinking…"):
        try:
            result = st.session_state.graph.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history,
                "output": "",
                "tools_used": [],
            })
            output: str          = result["output"]
            tools_used: List[str] = result.get("tools_used", [])
            st.session_state.chat_history = result.get(
                "chat_history", st.session_state.chat_history
            )
        except Exception as e:
            output     = f"⚠️ Error: {e}"
            tools_used = []

    st.session_state.tool_log.extend(tools_used)
    st.session_state.messages.append({
        "role": "assistant",
        "content": output,
        "tools_used": tools_used,
    })
    
    # Save assistant message to database if conversation saving is enabled
    if st.session_state.save_conversation and st.session_state.conversation_id:
        save_message(st.session_state.conversation_id, "assistant", output, tools_used)
    
    st.rerun()


# ── Tool log ──────────────────────────────────────────────────────────────────
if st.session_state.tool_log:
    with st.expander(f"🔧 Tool activity log ({len(st.session_state.tool_log)} calls)"):
        for i, t in enumerate(st.session_state.tool_log, 1):
            st.markdown(
                f"`{i}.` <span class='tool-badge'>⚡ {t}</span>",
                unsafe_allow_html=True,
            )
