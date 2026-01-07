from ai.research_agent.sources.sep import query_sep


def main() -> None:
    queries = ["social contract theory"]
    last_user_msg = "Explain social contract theory in philosophy."

    results = query_sep(queries, last_user_msg)

    print(f"Got {len(results)} QueryResult objects\n")
    for i, qr in enumerate(results[:5]):
        result = qr.get("result")
        print(result, "\n\n")


if __name__ == "__main__":
    main()

