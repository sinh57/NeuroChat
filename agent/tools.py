"""
agent/tools.py
Defines all LangChain tools available to the agent.
Each tool has full error handling so the agent never crashes on a bad tool call.
"""

import math
import json
import urllib.request
from datetime import datetime

from langchain.tools import tool


# ── Calculator ────────────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt, log, sin, cos, tan, pi, e, abs, round, floor, ceil.
    Also handles percentage: '15% of 847'.
    Example inputs: '15% of 847', 'sqrt(144) + 50', '(3**4) / 9'
    """
    try:
        expr = expression.strip().lower()

        # Handle "X% of Y"
        if "%" in expr and "of" in expr:
            parts = expr.replace("%", "").split("of")
            pct = float(parts[0].strip())
            value = float(parts[1].strip())
            result = (pct / 100) * value
            return f"{pct}% of {value} = {result:.4f}"

        expr = expr.replace("^", "**")
        safe_env = {
            "__builtins__": {},
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "abs": abs, "round": round, "pi": math.pi, "e": math.e,
            "pow": pow, "floor": math.floor, "ceil": math.ceil,
        }
        result = eval(expr, safe_env)  # noqa: S307
        return f"Result: {result}"
    except Exception as ex:
        return f"Calculator error: {ex}. Please check your expression."


# ── DateTime ──────────────────────────────────────────────────────────────────
@tool
def datetime_tool(query: str = "") -> str:
    """
    Returns current date, time, week number, and day of year.
    Use for any question about the current date or time.
    """
    now = datetime.now()
    return (
        f"Current Date : {now.strftime('%A, %B %d, %Y')}\n"
        f"Current Time : {now.strftime('%I:%M:%S %p')}\n"
        f"Week Number  : {now.isocalendar()[1]}\n"
        f"Day of Year  : {now.timetuple().tm_yday}"
    )


# ── Weather ───────────────────────────────────────────────────────────────────
@tool
def weather_tool(city: str) -> str:
    """
    Fetch live weather for any city using the free wttr.in API. No API key needed.
    Input: city name, e.g. 'London', 'New York', 'Tokyo'
    """
    try:
        safe_city = city.strip().replace(" ", "+")
        url = f"https://wttr.in/{safe_city}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "neurochat/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
        cur = data["current_condition"][0]
        return (
            f"Weather in {city}:\n"
            f"  Condition   : {cur['weatherDesc'][0]['value']}\n"
            f"  Temperature : {cur['temp_C']}°C / {cur['temp_F']}°F\n"
            f"  Feels Like  : {cur['FeelsLikeC']}°C\n"
            f"  Humidity    : {cur['humidity']}%\n"
            f"  Wind Speed  : {cur['windspeedKmph']} km/h"
        )
    except Exception as ex:
        return f"Could not fetch weather for '{city}': {ex}"


# ── Wikipedia ─────────────────────────────────────────────────────────────────
def _make_wikipedia_tool():
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=800)
        return WikipediaQueryRun(api_wrapper=wrapper)
    except Exception:
        return None


# ── Web Search ────────────────────────────────────────────────────────────────
def _make_search_tool():
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun()
    except ImportError:
        try:
            from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
            return DuckDuckGoSearchRun()
        except Exception:
            return None


# ── Tool Registry ─────────────────────────────────────────────────────────────
def get_tools(selected: list[str]) -> list:
    """
    Returns a list of LangChain tool objects for the given names.
    Skips any tool that fails to initialise (safe degradation).
    """
    registry = {
        "calculator": lambda: calculator,
        "datetime": lambda: datetime_tool,
        "weather": lambda: weather_tool,
        "wikipedia": _make_wikipedia_tool,
        "web_search": _make_search_tool,
    }

    tools = []
    for name in selected:
        if name in registry:
            t = registry[name]()
            if t is not None:
                tools.append(t)
    return tools
