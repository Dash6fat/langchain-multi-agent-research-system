try:
    from .pipeline import run_research_pipeline
except ImportError:
    from pipeline import run_research_pipeline


if __name__ == "__main__":
    topic = "The impact of AI on the job market in 2026"
    result = run_research_pipeline(topic)
    print("\n=== FINAL REPORT ===")
    print(result.get("report", "No report created."))
    if "critic" in result:
        print("\n=== CRITIC FEEDBACK ===")
        print(result["critic"])
