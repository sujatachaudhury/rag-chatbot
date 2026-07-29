import argparse

from src.evaluation import run_evaluation
from src.agent import ask_question
from src.vectorstore import ingest_directory


def main():
    parser = argparse.ArgumentParser(prog="ytrag", description="Agentic RAG over your own PDFs")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Chunk, embed, and store every PDF in data/pdfs")

    ask_parser = sub.add_parser("ask", help="Ask a question against the ingested corpus")
    ask_parser.add_argument("question")

    sub.add_parser("evaluate", help="Run the LangSmith evaluation suite against the agent")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_directory()
    elif args.command == "ask":
        result = ask_question(args.question)
        print(result["answer"])
    elif args.command == "evaluate":
        run_evaluation()


if __name__ == "__main__":
    main()
