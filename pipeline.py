try:
    from .agents import build_search_agent, build_reader_agent, writer_chain, critic_chain
except ImportError:
    from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


def run_research_pipeline(topic: str) -> dict:
    state = {}

    # Search agent working
    print("\n" + "=" * 50)
    print("step 1 - search agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result["messages"][-1].content
    print("\n search result", state["search_results"])

    print("\n" + "=" * 50)
    print("step 2 - reader agent is working...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [(
            "user",
            f"Based on the following search results about '{topic}', pick the most relevant URL and scrape it for deeper content.\n\nSearch Results:\n{state['search_results'][:800]}"
        )]
    })
    state["reader_results"] = reader_result["messages"][-1].content
    print("\n reader result", state["reader_results"])

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['reader_results']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })
    state["critic"] = critic_chain.invoke({
        "report": state["report"]
    })

    return state
