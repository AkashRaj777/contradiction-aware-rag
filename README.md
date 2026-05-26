# Contradiction-Aware RAG with Self-Corrective Retrieval

Hybrid-retrieval RAG over a synthetic company policy corpus. The pipeline grades evidence, rewrites queries when retrieval is weak, detects conflicting sources, refuses unanswerable questions, and checks answer faithfulness.

## Screenshots

### Home — ask any policy question
![Home view](docs/screenshots/01-home.png)

### Grounded answer with citations
![Grounded answer](docs/screenshots/02-answer.png)

### Conflicting sources flagged side-by-side
![Conflict detection](docs/screenshots/03-conflict.png)

### Calibrated refusal when evidence is missing
![Refusal](docs/screenshots/04-refuse.png)

## Features

- **Hybrid search:** BM25 + vector embeddings (Chroma + sentence-transformers)
- **Relevance grading:** cross-encoder reranker
- **Corrective loop:** query rewrite + retry when evidence is weak
- **Contradiction detection:** LLM judge across sources
- **Calibrated refusal:** no guessing when corpus lacks evidence
- **Faithfulness gate:** answer must match cited chunks
- **Eval suite:** 60 labeled questions (25 normal / 20 conflict / 15 refuse)

## Requirements

- **Python 3.11 or 3.12** (recommended: `py -3.12 -m venv .venv`)
- OpenAI API key

## Quick start

```bash
cd contradiction-rag
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
# Add OPENAI_API_KEY to .env

python scripts/generate_questions.py
python main.py ingest
python main.py ask "What is the refund window for standard purchases?"
```

## Evaluation

```bash
# Smoke test (10 questions) — uses API credits
python main.py eval --mode full --limit 10

# Compare naive baseline
python main.py eval --mode baseline --limit 10

# Full 60-question run
python main.py eval --mode full
```

Results saved to `data/eval/results_full.json` and `data/eval/results_baseline.json`.

## Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

## Project layout

```
contradiction-rag/
├── data/corpus/          # 11 synthetic policy markdown files
├── data/eval/            # questions.jsonl + eval results
├── src/                  # ingest, retrieve, pipeline, baseline
├── eval/run_eval.py
├── app/streamlit_app.py
└── main.py
```

## Corpus documents

| doc_id | Topic |
|--------|--------|
| password_policy_2024 / _2025 | Password rules (conflicting) |
| remote_work_precovid / _2025 | Remote vs hybrid (conflicting) |
| pto_policy_2024 / _2025 | PTO accrual (conflicting) |
| refund_policy_2024 / _2025_draft | Refunds (conflicting) |
| parental_leave_2025 | Leave |
| security_guidelines_2025 | Security |
| mfa_requirements_2025 | MFA |

## Eval results (60-question suite)

| Metric | Score |
|--------|-------|
| **Behavior accuracy** | **70.0%** (42/60) — correct answer vs conflict vs refuse |
| **Retrieval recall** | **82.2%** (37/45) — gold policy doc cited when applicable |

Details: `data/eval/results_full.json`

## Resume bullets

- Built a **corrective RAG pipeline** with hybrid BM25+vector retrieval, cross-encoder grading, and a **60-query** eval harness (25 normal / 20 conflict / 15 refuse) over a synthetic policy corpus.
- Implemented **contradiction-aware answers** and **calibrated refusal**, achieving **70% behavior accuracy** and **82% retrieval recall** on the labeled eval set.
- Shipped a **Streamlit demo** with citation-level grounding for policy Q&A (OpenAI `gpt-4o-mini`, local embeddings).

## API cost notes

Uses `gpt-4o-mini` for generation and judging. Run `--limit 10` while developing. Ingest and retrieval are **local** (no API cost).
