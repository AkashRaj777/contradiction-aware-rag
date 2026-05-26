"""
Streamlit demo for Contradiction-Aware RAG.
Run from project root: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components

from src.baseline import NaiveRAG
from src.pipeline import ContradictionRAGPipeline
from src.utils import load_eval_questions

# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Contradiction-Aware RAG",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ────────────────────────────────────────────────────────────
# IMPORTANT: the entire <style> block is on ONE LINE because Streamlit's
# markdown parser inserts paragraph breaks on blank lines inside long strings,
# which prematurely closes the <style> block and dumps CSS as visible text.
FONT_LINK = (
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Material+Symbols+Outlined" rel="stylesheet">'
    '<link href="https://fonts.googleapis.com/icon?'
    'family=Material+Icons" rel="stylesheet">'
)

CSS = (
    "<style>"
    ":root{--bg:#0e1117;--surface:#161b22;--surface-2:#1c222b;--text:#f1f5f9;"
    "--text-soft:#cbd5e1;--muted:#94a3b8;--border:#2a313c;--primary:#3b82f6;"
    "--primary-soft:rgba(59,130,246,0.18);}"
    "html,body,.stApp{background:#0e1117!important;"
    "font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif!important;"
    "color:#f1f5f9!important;font-size:17px;}"
    ".stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp label,.stApp span,"
    ".stApp div,.stMarkdown{font-family:'Inter',sans-serif!important;color:#f1f5f9;}"
    "div#MainMenu{visibility:hidden;}"
    "div footer{visibility:hidden;}"
    'div[data-testid="stToolbar"]{visibility:hidden;}'
    'div[data-testid="stDecoration"]{display:none;}'
    'div[data-testid="stSidebarCollapseButton"]{display:none!important;}'
    'div[data-testid="stSidebarCollapsedControl"]{display:none!important;}'
    'div[data-testid="stSidebarHeader"]{display:none!important;}'
    'button[data-testid="stBaseButton-header"]{display:none!important;}'
    'button[data-testid="stBaseButton-headerNoPadding"]{display:none!important;}'
    'section[data-testid="stSidebar"] button span{font-size:0!important;color:transparent!important;}'
    'div[data-testid="stExpanderToggleIcon"]{display:none!important;}'
    'details>summary [data-testid="stIconMaterial"]{display:none!important;}'
    'details>summary svg{display:none!important;}'
    'span.material-icons,span.material-icons-outlined,span.material-symbols-rounded,'
    'span.material-symbols-outlined{font-size:0!important;color:transparent!important;'
    'width:0!important;height:0!important;overflow:hidden!important;display:none!important;}'
    'details>summary{padding-left:1rem!important;}'
    'section[data-testid="stSidebar"] details>summary{position:relative!important;'
    'padding-left:1.6rem!important;list-style:none!important;cursor:pointer!important;}'
    'section[data-testid="stSidebar"] details>summary::-webkit-details-marker{display:none!important;}'
    'section[data-testid="stSidebar"] details>summary{padding-left:2.2rem!important;}'
    'section[data-testid="stSidebar"] details>summary::before{content:"\\25B8"!important;'
    'position:absolute!important;left:0.5rem!important;top:50%!important;'
    'transform:translateY(-50%)!important;color:#3b82f6!important;font-size:1.8rem!important;'
    'line-height:1!important;font-weight:900!important;'
    'transition:transform 0.2s ease,color 0.2s ease!important;'
    'display:inline-block!important;transform-origin:center!important;'
    'text-shadow:0 0 8px rgba(59,130,246,0.5)!important;}'
    'section[data-testid="stSidebar"] details[open]>summary::before{'
    'transform:translateY(-50%) rotate(90deg)!important;color:#60a5fa!important;}'
    "h1.hero-title{font-size:3rem!important;font-weight:800!important;"
    "color:#f1f5f9!important;letter-spacing:-0.03em;"
    "margin:0.5rem 0 0.6rem 0!important;line-height:1.1!important;}"
    "p.hero-sub{font-size:1.2rem!important;color:#cbd5e1!important;"
    "line-height:1.65!important;max-width:none!important;width:100%!important;"
    "margin:0 0 1.75rem 0!important;font-weight:400!important;}"
    "div.card{background:#161b22;border:1px solid #2a313c;border-radius:14px;"
    "padding:1.6rem 1.75rem;margin-bottom:1.25rem;"
    "box-shadow:0 4px 24px rgba(0,0,0,0.35);}"
    "div.card h3{font-size:1.4rem!important;font-weight:700!important;"
    "color:#f1f5f9!important;margin:0 0 1rem 0!important;letter-spacing:-0.01em;}"
    "div.card h4{font-size:1.15rem!important;font-weight:700!important;"
    "color:#f1f5f9!important;margin:1.25rem 0 0.5rem 0!important;}"
    "div.card p,div.card li{font-size:1.05rem!important;color:#cbd5e1!important;"
    "line-height:1.65!important;}"
    "div.steps-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1.4rem;}"
    "div.step-card{background:#1c222b;border:1px solid #2a313c;border-radius:12px;"
    "padding:1.3rem 1.25rem;}"
    "div.step-num{display:inline-flex;align-items:center;justify-content:center;"
    "width:2.4rem;height:2.4rem;border-radius:10px;background:#3b82f6;"
    "color:white;font-weight:700;font-size:1.1rem;margin-bottom:0.85rem;"
    "font-family:'Inter',sans-serif!important;}"
    "div.step-title{font-weight:700;font-size:1.1rem;color:#f1f5f9;"
    "margin-bottom:0.4rem;letter-spacing:-0.01em;}"
    "div.step-desc{font-size:1rem;color:#94a3b8;line-height:1.6;}"
    "div.flowchart{display:flex;align-items:stretch;justify-content:space-between;"
    "gap:0.5rem;flex-wrap:nowrap;margin-top:0.5rem;}"
    "div.flow-step{flex:1;padding:1rem 0.85rem;border-radius:12px;"
    "display:flex;flex-direction:column;align-items:center;justify-content:center;"
    "text-align:center;min-height:5.5rem;color:white;font-weight:600;}"
    "div.flow-step .fs-num{font-size:0.78rem;font-weight:700;opacity:0.85;"
    "letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.35rem;}"
    "div.flow-step .fs-label{font-size:1rem;font-weight:700;line-height:1.3;}"
    "div.fs-1{background:linear-gradient(135deg,#3b82f6,#2563eb);}"
    "div.fs-2{background:linear-gradient(135deg,#8b5cf6,#7c3aed);}"
    "div.fs-3{background:linear-gradient(135deg,#ec4899,#db2777);}"
    "div.fs-4{background:linear-gradient(135deg,#f59e0b,#d97706);}"
    "div.fs-5{background:linear-gradient(135deg,#10b981,#059669);}"
    "div.flow-arrow{display:flex;align-items:center;justify-content:center;"
    "color:#64748b;font-size:1.6rem;font-weight:700;flex:0 0 auto;padding:0 0.15rem;}"
    "span.decision-answer{background:rgba(16,185,129,0.15);color:#6ee7b7;"
    "border:1px solid rgba(16,185,129,0.4);}"
    "span.decision-conflict{background:rgba(245,158,11,0.15);color:#fcd34d;"
    "border:1px solid rgba(245,158,11,0.4);}"
    "span.decision-refuse{background:rgba(239,68,68,0.15);color:#fca5a5;"
    "border:1px solid rgba(239,68,68,0.4);}"
    "span.decision-badge{display:inline-block;padding:0.55rem 1.2rem;"
    "border-radius:8px;font-size:1.05rem;font-weight:700;text-transform:uppercase;"
    "letter-spacing:0.04em;}"
    "p.metric-label{font-size:0.85rem!important;font-weight:600!important;"
    "color:#94a3b8!important;text-transform:uppercase;letter-spacing:0.06em;"
    "margin-bottom:0.4rem!important;}"
    "p.metric-value{font-size:1.5rem!important;font-weight:700!important;"
    "color:#f1f5f9!important;}"
    "span.cite-chip{display:inline-block;background:rgba(59,130,246,0.18);"
    "color:#93c5fd;padding:0.35rem 0.75rem;border-radius:6px;font-size:0.92rem;"
    "font-weight:600;margin:0.25rem 0.3rem 0.25rem 0;"
    "border:1px solid rgba(59,130,246,0.3);}"
    'section[data-testid="stSidebar"]{'
    "background:linear-gradient(180deg,#0f172a 0%,#1e293b 100%);}"
    'section[data-testid="stSidebar"] *{color:#e2e8f0!important;'
    "font-family:'Inter',sans-serif!important;}"
    'section[data-testid="stSidebar"] h3{color:#fff!important;font-weight:700!important;'
    "font-size:1.15rem!important;margin-bottom:0.75rem!important;}"
    'section[data-testid="stSidebar"] label{font-size:1rem!important;font-weight:500!important;}'
    'div.stButton>button[kind="primary"]{'
    "background:linear-gradient(135deg,#2563eb,#3b82f6)!important;border:none!important;"
    "font-weight:700!important;font-size:1.1rem!important;padding:0.75rem 1.75rem!important;"
    "border-radius:10px!important;box-shadow:0 4px 14px rgba(37,99,235,0.35)!important;"
    "color:white!important;}"
    "div.stTextArea textarea{font-size:1.1rem!important;border-radius:10px!important;"
    "border:1px solid #2a313c!important;background:#1c222b!important;"
    "color:#f1f5f9!important;padding:1rem!important;}"
    'div[data-testid="stVerticalBlockBorderWrapper"]{'
    "background:#1c222b!important;border-color:#2a313c!important;}"
    "</style>"
)

# Inject font link and CSS as a single line each (markdown-safe)
st.markdown(FONT_LINK, unsafe_allow_html=True)
st.markdown(CSS, unsafe_allow_html=True)

# JS: hide sidebar collapse button — CSS selectors vary by Streamlit version,
# so we find and hide by text content instead (runs in parent frame via iframe).
components.html(
    """
    <script>
    (function() {
        // Matches Material icon names like "keyboard_arrow_right", "expand_more"
        var ICON_RE = /^[a-z][a-z_]*[a-z]$/;
        var ICON_WORDS = ['arrow', 'keyboard', 'chevron', 'expand', 'collapse',
                          'menu', 'close', 'check'];

        function isIconName(t) {
            if (!t || t.length < 4 || t.length > 40) return false;
            if (t.indexOf(' ') !== -1) return false;
            if (!ICON_RE.test(t)) return false;
            return ICON_WORDS.some(function(w) { return t.indexOf(w) !== -1; });
        }

        function fix() {
            try {
                var doc = window.parent.document;
                if (!doc || !doc.body) return;

                // Walk every leaf element in the WHOLE PAGE, not just sidebar
                var all = doc.body.querySelectorAll('*');
                all.forEach(function(el) {
                    if (el.children.length > 0) return;
                    var t = (el.innerText || el.textContent || '').trim();
                    if (isIconName(t)) {
                        el.style.fontSize = '0';
                        el.style.color = 'transparent';
                        el.style.width = '0';
                        el.style.height = '0';
                        el.style.overflow = 'hidden';
                        el.style.display = 'none';
                    }
                });

                // Hide all Streamlit expander toggle icons (chevrons)
                doc.querySelectorAll('[data-testid="stExpanderToggleIcon"]').forEach(function(el){
                    el.style.display = 'none';
                });
                doc.querySelectorAll('details>summary [data-testid="stIconMaterial"]').forEach(function(el){
                    el.style.display = 'none';
                });
                doc.querySelectorAll('details>summary svg').forEach(function(el){
                    el.style.display = 'none';
                });

                // Also catch the page-level sidebar header chrome
                var hdr = doc.querySelector('[data-testid="stSidebarHeader"]');
                if (hdr) hdr.style.display = 'none';
                var cb = doc.querySelector('[data-testid="stSidebarCollapseButton"]');
                if (cb) cb.style.display = 'none';
            } catch(e) {}
        }
        fix();
        setTimeout(fix, 200);
        setTimeout(fix, 600);
        setTimeout(fix, 1500);
        setTimeout(fix, 3000);

        // Also re-run whenever the DOM changes
        try {
            var obs = new MutationObserver(fix);
            obs.observe(window.parent.document.body, { childList: true, subtree: true });
        } catch(e) {}
    })();
    </script>
    """,
    height=0,
)

DECISION_STYLES = {
    "answer": "decision-answer",
    "conflict": "decision-conflict",
    "refuse": "decision-refuse",
}

DECISION_LABELS = {
    "answer": "Grounded answer",
    "conflict": "Conflicting sources",
    "refuse": "Insufficient evidence",
}


@st.cache_resource
def load_full_pipeline():
    return ContradictionRAGPipeline()


@st.cache_resource
def load_baseline():
    return NaiveRAG()


def render_decision_badge(decision: str) -> None:
    css = DECISION_STYLES.get(decision, "decision-answer")
    label = DECISION_LABELS.get(decision, decision)
    st.markdown(
        f'<span class="decision-badge {css}">{label}</span>',
        unsafe_allow_html=True,
    )


def render_citations(citations: list[str]) -> None:
    if not citations:
        return
    chips = "".join(f'<span class="cite-chip">{c}</span>' for c in citations)
    st.markdown(f"**Sources cited**<br>{chips}", unsafe_allow_html=True)


# ── Sidebar (controls only) ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    mode = st.radio(
        "Pipeline mode",
        ["Full (contradiction-aware)", "Baseline (naive RAG)"],
        help="Full: grading, rewrite, conflict detection, refusal. Baseline: retrieve + answer only.",
    )
    show_samples = st.checkbox("Show sample questions", value=True)
    samples = load_eval_questions() if show_samples else []
    pick = "—"
    if show_samples and samples:
        labels = [f"{q['id']} · {q['bucket']}: {q['question'][:55]}…" for q in samples[:15]]
        pick = st.selectbox("Try a sample", ["—"] + labels)

    st.markdown("---")
    with st.expander("Eval benchmark", expanded=False):
        st.markdown(
            """
            **60-question suite** (local eval):
            - **70%** behavior accuracy
            - **82%** retrieval recall
            """
        )

# ── Main header ──────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">Contradiction-Aware RAG</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Ask a question about company policies. The system reads the documents, '
    "gives you the answer with sources, warns you when policies disagree, and refuses to guess "
    "when the answer is not in the documents.</p>",
    unsafe_allow_html=True,
)

# ── Question input ───────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Ask a question")
question = st.text_area(
    "Question",
    height=120,
    placeholder="e.g. What is the refund window for standard purchases?",
    label_visibility="collapsed",
)
if show_samples and pick != "—":
    idx = [f"{q['id']} · {q['bucket']}: {q['question'][:55]}…" for q in samples[:15]].index(pick)
    question = samples[idx]["question"]

run = st.button("Run analysis", type="primary", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

# ── Input sanity check (B) ───────────────────────────────────────────────────
# Reject obvious junk inputs ("hi", "ok", single words, etc.) BEFORE we hit the
# pipeline. This saves OpenAI credits on garbage queries that would just refuse
# anyway, and avoids the confusing "company policy documents" rewrite leak.
MIN_QUESTION_LEN = 8          # at least 8 non-space characters
MIN_QUESTION_WORDS = 3        # at least 3 words
GREETING_BLACKLIST = {
    "hi", "hello", "hey", "yo", "sup", "hola", "test", "testing",
    "ok", "okay", "thanks", "thank you", "bye", "?", "??", "???",
}

def is_garbage_question(q: str) -> bool:
    """True if the input is clearly not a real policy question."""
    clean = q.strip().lower().rstrip("?.! ")
    if len(clean.replace(" ", "")) < MIN_QUESTION_LEN:
        return True
    if len(clean.split()) < MIN_QUESTION_WORDS:
        return True
    if clean in GREETING_BLACKLIST:
        return True
    return False


# ── Results ──────────────────────────────────────────────────────────────────
if run:
    if not question.strip():
        st.warning("Please enter a question before running analysis.")
    # (B) Catch junk inputs before they reach the LLM. Friendly message instead
    # of a confusing refusal with a generic "company policy documents" rewrite.
    elif is_garbage_question(question):
        st.warning(
            "Please enter a real policy question (at least a full sentence). "
            "Examples: *'What is the refund window?'* or *'Can I work remotely from another country?'*"
        )
    else:
        with st.spinner("Retrieving evidence and generating response…"):
            is_full = mode.startswith("Full")
            if is_full:
                result = load_full_pipeline().run(question.strip(), log=True)
            else:
                result = load_baseline().run(question.strip())
                result.decision = "answer"
                result.confidence = "n/a"

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Response")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<p class="metric-label">Decision</p>', unsafe_allow_html=True)
            render_decision_badge(result.decision)
        with m2:
            st.markdown('<p class="metric-label">Confidence</p>', unsafe_allow_html=True)
            conf = getattr(result, "confidence", "—")
            st.markdown(f'<p class="metric-value">{conf}</p>', unsafe_allow_html=True)
        with m3:
            st.markdown('<p class="metric-label">Pipeline</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="metric-value">{"Full" if is_full else "Baseline"}</p>',
                unsafe_allow_html=True,
            )

        # (A) Only surface the corrective rewrite when it actually helped —
        # i.e. the pipeline produced a real answer or flagged a conflict.
        # On refusals the rewrite is just the LLM's last-ditch guess (e.g.
        # "company policy documents" for "hi") and shouldn't be shown to users.
        if (
            is_full
            and getattr(result, "rewritten_query", None)
            and result.decision != "refuse"
        ):
            st.info(f"**Corrective query rewrite:** {result.rewritten_query}")

        if is_full and getattr(result, "contradiction_explanation", None):
            st.warning(f"**Contradiction note:** {result.contradiction_explanation}")

        st.markdown("#### Answer")
        with st.container(border=True):
            st.markdown(result.answer)

        citations = getattr(result, "citations", []) or []
        if citations:
            st.markdown("<br>", unsafe_allow_html=True)
            render_citations(citations)

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Technical details", expanded=False):
            if is_full:
                st.json(
                    {
                        "decision": result.decision,
                        "confidence": result.confidence,
                        "citations": citations,
                        "rewritten_query": getattr(result, "rewritten_query", None),
                        "retries": getattr(result, "retries", 0),
                        "metadata": getattr(result, "metadata", {}),
                    }
                )
            else:
                st.json({"decision": "answer", "citations": citations})

# ── How to use (4-column grid, full width) ───────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

steps_html = (
    '<div class="card">'
    "<h3>How to use this app</h3>"
    '<div class="steps-grid">'
    '<div class="step-card">'
    '<div class="step-num">1</div>'
    '<div class="step-title">Pick a mode</div>'
    '<div class="step-desc">Use <strong>Full</strong> for the complete system. '
    "Choose <strong>Baseline</strong> to compare against a simple RAG.</div>"
    "</div>"
    '<div class="step-card">'
    '<div class="step-num">2</div>'
    '<div class="step-title">Ask a question</div>'
    '<div class="step-desc">Type your own policy question, or pick one from the '
    "<strong>sample</strong> dropdown in the sidebar.</div>"
    "</div>"
    '<div class="step-card">'
    '<div class="step-num">3</div>'
    '<div class="step-title">Run analysis</div>'
    '<div class="step-desc">The first run takes <strong>1–2 minutes</strong> while '
    "models load. Every run after that is fast.</div>"
    "</div>"
    '<div class="step-card">'
    '<div class="step-num">4</div>'
    '<div class="step-title">Read the result</div>'
    "<div class=\"step-desc\">You'll see a <strong>decision</strong>, the "
    "<strong>answer</strong>, the <strong>sources</strong> used, and any extra notes.</div>"
    "</div>"
    "</div>"
    "</div>"
)
st.markdown(steps_html, unsafe_allow_html=True)

# ── Pipeline flowchart (colored, horizontal, with arrows) + What you get back ─
flow_col, output_col = st.columns([1.4, 1.0], gap="large")

with flow_col:
    flow_html = (
        '<div class="card">'
        "<h3>What happens behind the scenes</h3>"
        '<div class="flowchart">'
        '<div class="flow-step fs-1">'
        '<div class="fs-num">Step 1</div>'
        '<div class="fs-label">Your<br>question</div>'
        "</div>"
        '<div class="flow-arrow">→</div>'
        '<div class="flow-step fs-2">'
        '<div class="fs-num">Step 2</div>'
        '<div class="fs-label">Find<br>chunks</div>'
        "</div>"
        '<div class="flow-arrow">→</div>'
        '<div class="flow-step fs-3">'
        '<div class="fs-num">Step 3</div>'
        '<div class="fs-label">Score &<br>retry</div>'
        "</div>"
        '<div class="flow-arrow">→</div>'
        '<div class="flow-step fs-4">'
        '<div class="fs-num">Step 4</div>'
        '<div class="fs-label">Detect<br>conflicts</div>'
        "</div>"
        '<div class="flow-arrow">→</div>'
        '<div class="flow-step fs-5">'
        '<div class="fs-num">Step 5</div>'
        '<div class="fs-label">Answer or<br>refuse</div>'
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(flow_html, unsafe_allow_html=True)

with output_col:
    output_html = (
        '<div class="card">'
        "<h3>What you get back</h3>"
        "<ul>"
        "<li><strong>Decision</strong> — a clear answer, both sides of a conflict, or a refusal</li>"
        "<li><strong>Confidence</strong> — how sure the system is about the result</li>"
        "<li><strong>Answer text</strong> — written using real policy documents</li>"
        "<li><strong>Sources</strong> — the policy file names that were used</li>"
        "<li><strong>Notes</strong> — only shown if the query was rewritten or sources clashed</li>"
        "</ul>"
        "</div>"
    )
    st.markdown(output_html, unsafe_allow_html=True)
