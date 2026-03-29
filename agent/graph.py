"""
agent/graph.py
Defines the LangGraph StateGraph for the conversational agent.

Key design decisions:
- Memory object lives OUTSIDE the graph so it persists across Streamlit reruns.
- AgentExecutor is cached alongside the graph (same object, no re-init).
- State carries only serialisable data (dicts / lists / strings).
"""

from __future__ import annotations

from typing import Any, TypedDict

# AgentExecutor moved to langchain.agents.agent in newer versions
try:
    from langchain.agents import AgentExecutor
except ImportError:
    from langchain.agents.agent import AgentExecutor

# create_openai_tools_agent moved to langchain.agents in some, langchain_core in others
try:
    from langchain.agents import create_openai_tools_agent
except ImportError:
    from langchain_core.agents import create_openai_tools_agent

# Memory classes moved to langchain_community in LangChain >=0.3
try:
    from langchain.memory import (
        ConversationBufferMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryMemory,
    )
except ImportError:
    from langchain_community.memory import (  # type: ignore
        ConversationBufferMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryMemory,
    )

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agent.tools import get_tools


# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NeuroChat, an advanced conversational AI assistant.

You have:
- Persistent memory of this entire conversation
- Access to tools: web search, calculator, Wikipedia, date/time, and weather

Guidelines:
- Think step-by-step before answering
- When using a tool, briefly state why before calling it
- Summarise tool results clearly — don't dump raw output
- Be concise and accurate
- Use markdown formatting for code, lists, and tables
- If a tool fails, handle it gracefully and answer from your own knowledge
"""


# ── LangGraph State ───────────────────────────────────────────────────────────
# Use typing.List / typing.Dict (not built-in list/dict) so LangGraph
# can infer types correctly on Python 3.8 – 3.11
from typing import List, Dict

class AgentState(TypedDict):
    input: str                              # latest user message
    chat_history: List[Dict[str, str]]      # serialised history [{role, content}]
    output: str                             # agent reply
    tools_used: List[str]                   # tool names called this turn


# ── Agent node ────────────────────────────────────────────────────────────────
def make_agent_node(executor: AgentExecutor, memory: Any):
    """
    Returns a node function closed over a *single* executor + memory instance.
    This is the fix for the memory-reset bug: both objects are created once and
    reused for the lifetime of the Streamlit session.
    """

    def agent_node(state: AgentState) -> AgentState:
        user_input = state["input"]

        # Restore memory from serialised history on first call after a rerun
        stored = state.get("chat_history", [])
        if stored and not memory.chat_memory.messages:
            memory.chat_memory.clear()
            for entry in stored:
                if entry["role"] == "human":
                    memory.chat_memory.add_user_message(entry["content"])
                else:
                    memory.chat_memory.add_ai_message(entry["content"])

        # Run the agent
        try:
            result = executor.invoke({
                "input": user_input,
                "chat_history": memory.chat_memory.messages,
            })
            output = result.get("output", "")
            steps = result.get("intermediate_steps", [])
        except Exception as ex:
            output = f"⚠️ Agent error: {ex}"
            steps = []

        # Extract tool names from intermediate steps
        tools_used: list[str] = []
        for action, _ in steps:
            name = getattr(action, "tool", None)
            if name and name not in tools_used:
                tools_used.append(name)

        # Persist turn to memory
        memory.save_context({"input": user_input}, {"output": output})

        # Serialise updated history back into state
        serialised = [
            {
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": m.content,
            }
            for m in memory.chat_memory.messages
        ]

        return {
            "input": user_input,
            "chat_history": serialised,
            "output": output,
            "tools_used": tools_used,
        }

    return agent_node


# ── Graph factory ─────────────────────────────────────────────────────────────
def build_agent(
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    selected_tools: List[str] = None,
    memory_type: str = "ConversationBuffer",
    window_k: int = 5,
) -> tuple:
    """
    Build and compile a LangGraph agent.

    Returns (compiled_graph, memory) so Streamlit can store both in
    session_state and reuse them across reruns without re-initialising.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    # Memory
    if memory_type == "ConversationSummary":
        memory = ConversationSummaryMemory(
            llm=llm, memory_key="chat_history", return_messages=True
        )
    elif memory_type == "ConversationWindow":
        memory = ConversationBufferWindowMemory(
            k=window_k, memory_key="chat_history", return_messages=True
        )
    else:
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    # Tools
    tool_objects = get_tools(selected_tools or [])

    # Prompt — must include agent_scratchpad placeholder for tool calling
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tool_objects, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tool_objects,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6,
        return_intermediate_steps=True,
    )

    # LangGraph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", make_agent_node(executor, memory))
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)

    return workflow.compile(), memory
