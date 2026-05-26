"""
CLI entry point for the project.

Examples:
  python main.py ingest
  python main.py eval --mode full --limit 10
  python main.py eval --mode baseline --limit 60
  python main.py ask "What is the minimum password length?"
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_ingest(_: argparse.Namespace) -> None:
    from src.ingest import run_ingest

    run_ingest()


def cmd_gen_eval(_: argparse.Namespace) -> None:
    from scripts.generate_questions import main as gen_questions

    gen_questions()


def cmd_eval(args: argparse.Namespace) -> None:
    from eval.run_eval import run_evaluation

    run_evaluation(mode=args.mode, limit=args.limit)


def cmd_ask(args: argparse.Namespace) -> None:
    if args.mode == "baseline":
        from src.baseline import NaiveRAG

        result = NaiveRAG().run(args.question)
        print(f"\n[baseline / {result.decision}]\n{result.answer}\n")
        print("Citations:", result.citations)
    else:
        from src.pipeline import ContradictionRAGPipeline

        result = ContradictionRAGPipeline().run(args.question, log=True)
        print(f"\n[{result.decision} / confidence={result.confidence}]\n{result.answer}\n")
        print("Citations:", result.citations)
        if result.rewritten_query:
            print("Rewritten query:", result.rewritten_query)


def main() -> None:
    parser = argparse.ArgumentParser(description="Contradiction-Aware RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Build Chroma index from data/corpus")
    p_ingest.set_defaults(func=cmd_ingest)

    p_gen = sub.add_parser("gen-eval", help="Regenerate questions.jsonl")
    p_gen.set_defaults(func=cmd_gen_eval)

    p_eval = sub.add_parser("eval", help="Run evaluation suite")
    p_eval.add_argument("--mode", choices=["full", "baseline"], default="full")
    p_eval.add_argument("--limit", type=int, default=0, help="0 = all questions")
    p_eval.set_defaults(func=cmd_eval)

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--mode", choices=["full", "baseline"], default="full")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
