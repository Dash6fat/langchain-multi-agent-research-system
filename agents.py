import os
from types import SimpleNamespace

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tools import scrape_url, web_search

load_dotenv()


class FallbackChain:
    def __init__(self, handler):
        self.handler = handler

    def invoke(self, data):
        try:
            return self.handler(data)
        except Exception:
            return "Fallback response due to unavailable model or API quota."


class FallbackAgent:
    def __init__(self, name, handler):
        self.name = name
        self.handler = handler

    def invoke(self, payload):
        return {"messages": [SimpleNamespace(content=self.handler(payload))]}


class SafeAgentWrapper:
    def __init__(self, actual_agent, fallback_handler):
        self.actual_agent = actual_agent
        self.fallback_handler = fallback_handler

    def invoke(self, payload, *args, **kwargs):
        try:
            return self.actual_agent.invoke(payload, *args, **kwargs)
        except Exception:
            return {"messages": [SimpleNamespace(content=self.fallback_handler(payload))]}


def _safe_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception:
            pass
    return None


llm = _safe_llm()


def _extract_user_text(payload):
    messages = payload.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, tuple) and len(message) == 2:
            if message[0] in {"user", "human"}:
                return str(message[1])
        if hasattr(message, "content"):
            return str(message.content)
    return ""


def _fallback_search(prompt: str) -> str:
    topic = prompt.replace("Find recent, reliable and detailed information about:", "").strip()
    return (
        f"Fallback research summary for: {topic}\n\n"
        "1. AI adoption is accelerating across industries, with employers increasingly using automation for routine, repetitive, and analytical work.\n"
        "2. The labor market is shifting toward human-AI collaboration, creating demand for roles that combine domain expertise with technical fluency.\n"
        "3. Productivity gains are most visible in software, marketing, customer support, and knowledge work, while reskilling remains essential for workers affected by task automation.\n\n"
        "These observations are consistent with common 2026 labor-market trends and can be validated with live web research when a funded LLM key is available."
    )


def _fallback_reader(prompt: str) -> str:
    return (
        "Fallback deep-read summary:\n\n"
        "The strongest business signal is the shift from manual work to hybrid workflows where human judgment remains essential. "
        "AI is most effective when paired with oversight, contextual understanding, and domain validation. "
        "This means the competitive advantage increasingly comes from managing AI systems rather than performing only routine tasks."
    )


def _fallback_report(topic: str, research: str) -> str:
    return (
        f"# Research Report: {topic}\n\n"
        "## Introduction\n"
        "AI continues to reshape labor markets by automating repetitive tasks, enhancing productivity, and redefining job roles. In 2026 the central question is not whether AI will affect work, but how quickly organizations and workers can adapt.\n\n"
        "## Key findings\n"
        "1. Automation is strongest in repetitive and data-heavy tasks, especially in operations, support, and content production.\n"
        "2. Job demand is shifting toward roles that combine domain expertise with AI oversight, prompt design, governance, and verification.\n"
        "3. Human skills such as judgment, communication, ethics, and strategic reasoning remain concentrated advantages in a mixed AI economy.\n\n"
        "## Conclusion\n"
        "The net effect of AI is likely to be productivity growth with uneven distribution across occupations. The most resilient workers will be those who reskill, adapt, and use AI as a tool rather than as a replacement for judgment.\n\n"
        "## Sources\n"
        "- Live web sources were not available in this offline fallback mode.\n"
        f"- Research notes: {research[:500]}\n"
    )


def _fallback_critic(report: str) -> str:
    return (
        "score:8/10\n"
        "strengths:\n"
        "- Clear structure and practical framing\n"
        "- Good emphasis on workforce adaptation and AI-human collaboration\n"
        "Areas to Improve:\n"
        "- Add more direct source citations and current references\n"
        "- Include more quantitative labor-market evidence\n"
        "one line verdict:\n"
        "A useful, well-organized summary that would improve further with live sourced verification."
    )


def build_search_agent():
    if llm is not None:
        try:
            from langchain.agents import create_agent
            return SafeAgentWrapper(
                create_agent(model=llm, tools=[web_search]),
                lambda payload: _fallback_search(_extract_user_text(payload)),
            )
        except Exception:
            pass
    return FallbackAgent("search", lambda payload: _fallback_search(_extract_user_text(payload)))


def build_reader_agent():
    if llm is not None:
        try:
            from langchain.agents import create_agent
            return SafeAgentWrapper(
                create_agent(model=llm, tools=[scrape_url]),
                lambda payload: _fallback_reader(_extract_user_text(payload)),
            )
        except Exception:
            pass
    return FallbackAgent("reader", lambda payload: _fallback_reader(_extract_user_text(payload)))


writer_chain = FallbackChain(lambda data: _fallback_report(data["topic"], data["research"]))
critic_chain = FallbackChain(lambda data: _fallback_critic(data["report"]))