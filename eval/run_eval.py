"""
Run the 60-question eval set and print behavior + retrieval metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.baseline import NaiveRAG
from src.pipeline import ContradictionRAGPipeline
from src.utils import load_eval_questions


def _behavior_match(expected: str, actual: str) -> bool:
    """Map pipeline decisions to eval expected_behavior."""
    if expected == "refuse":
        return actual == "refuse"
    if expected == "present_both_sides":
        return actual == "conflict"
    if expected == "answer":
        return actual == "answer"
    return False


def _retrieval_hit(gold_doc_ids: list[str], cited: list[str]) -> bool:
    """Recall@5 proxy: any gold document appeared in citations."""
    if not gold_doc_ids:
        return False
    cited_set = set(cited)
    return any(g in cited_set for g in gold_doc_ids)


def _log(msg: str) -> None:
    """Print immediately (Windows buffers stdout when not attached to a TTY)."""
    print(msg, flush=True)


def run_evaluation(mode: str = "full", limit: int = 0) -> dict:
    questions = load_eval_questions()
    if limit > 0:
        questions = questions[:limit]

    _log(f"Loading pipeline ({mode})... models may take 30-60s on first run.")
    if mode == "baseline":
        runner = NaiveRAG()
    else:
        runner = ContradictionRAGPipeline()
    _log(f"Pipeline ready. Running {len(questions)} questions (~1-2 min each).\n")

    behavior_correct = 0
    retrieval_hits = 0
    retrieval_total = 0
    results = []
    out_path = ROOT / "data" / "eval" / f"results_{mode}.json"

    for i, q in enumerate(questions, 1):
        _log(f"[{i}/{len(questions)}] {q['id']} ({q['bucket']}) — calling OpenAI...")
        if mode == "baseline":
            out = runner.run(q["question"])
            actual_decision = "answer"  # baseline always answers
            citations = out.citations
            answer = out.answer
        else:
            out = runner.run(q["question"], log=False)
            actual_decision = out.decision
            citations = out.citations
            answer = out.answer

        expected = q["expected_behavior"]
        ok = _behavior_match(expected, actual_decision)
        if ok:
            behavior_correct += 1

        if q["gold_doc_ids"]:
            retrieval_total += 1
            if _retrieval_hit(q["gold_doc_ids"], citations):
                retrieval_hits += 1

        row = {
            "id": q["id"],
            "bucket": q["bucket"],
            "expected": expected,
            "actual": actual_decision,
            "behavior_ok": ok,
            "citations": citations,
        }
        results.append(row)
        _log(f"    -> {actual_decision} (expected {expected}) {'OK' if ok else 'MISS'}")

        # Save partial results so a long run is not lost
        partial = {
            "summary": {"completed": i, "total": len(questions), "mode": mode},
            "results": results,
        }
        out_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    n = len(questions)
    summary = {
        "mode": mode,
        "total": n,
        "behavior_accuracy": round(100 * behavior_correct / n, 1),
        "retrieval_recall_pct": round(100 * retrieval_hits / retrieval_total, 1)
        if retrieval_total
        else 0.0,
        "behavior_correct": behavior_correct,
        "retrieval_hits": retrieval_hits,
        "retrieval_total": retrieval_total,
    }

    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )

    _log("\n=== Eval Summary ===")
    _log(json.dumps(summary, indent=2))
    _log(f"Detailed results saved to {out_path}")
    return summary


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "baseline"], default="full")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    run_evaluation(mode=args.mode, limit=args.limit)
