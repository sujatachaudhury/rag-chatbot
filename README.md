# YTRAG

An agentic RAG chatbot that answers questions about your own PDFs. Built
this mainly to move past "naive RAG" (embed, retrieve top-k, stuff into a
prompt) and actually implement the retrieve -> grade -> retry loop with
LangGraph.

The interesting bit is that the agent doesn't just trust whatever it
retrieves. It asks an LLM "does this context actually answer the question?",
and if the answer is no, it rewrites the search query and tries again (up to
a couple of times) before giving up and answering from general knowledge
instead of just hallucinating from bad context.

```
retrieve -> grade -> relevant? -- yes --> generate
               ^                |
               |               no
               |                v
             rewrite query <----
             (max MAX_QUERY_REWRITES times, then generate anyway)
```

See `src/agent.py` for the actual graph.

## What's in here

- `app.py` — Streamlit UI. Upload a PDF, ask questions, see the sources it used.
- `main.py` — CLI with `ingest` / `ask` / `evaluate` subcommands.
- `src/ingestion.py` — loads PDFs page by page, chunks them.
- `src/embeddings.py` — wraps `sentence-transformers` for embedding text.
- `src/vectorstore.py` — Chroma persistence + the ingest pipeline.
- `src/retriever.py` — embeds the query, does cosine similarity search.
- `src/agent.py` — the LangGraph agent described above.
- `src/evaluation.py` — LangSmith eval harness (correctness + groundedness).
- `src/config.py` — everything tunable in one place.
- `data/pdfs/` — drop your PDFs here.
- `data/vector_store/` — Chroma's persisted DB, gets created on first ingest.
- `notebook/` — scratch notebooks from when I was building this out.

## Setup

Needs Python 3.13+. I used `uv`, but pip works too:

```bash
uv sync
# or
pip install -r requirements.txt
```

You'll need a `.env` file with:

```
GROQ_API_KEY=...
GOOGLE_API_KEY=...
```

Groq runs the actual agent (Llama 3.3 70B). Google's key is only used by
`evaluation.py` for the Gemini grader — skip it if you're not running evals.

## Running it

Put some PDFs in `data/pdfs/` and ingest them:

```bash
python main.py ingest
```

Then either ask from the terminal:

```bash
python main.py ask "what's this paper about?"
```

or launch the Streamlit app, which also lets you upload PDFs straight from
the sidebar:

```bash
streamlit run app.py
```

Answers that were actually grounded in your documents show a "Sources"
expander with the file name and similarity score. If nothing relevant turned
up in the corpus, it says so instead of pretending.

## Evaluation

`python main.py evaluate` runs a LangSmith eval measuring correctness against
ground truth and whether answers stay grounded in what was retrieved.
`EXAMPLES` in `src/evaluation.py` has 6 question/answer pairs grounded in
`data/pdfs/Denoising_Report.pdf`.

Latest run (Groq `llama-3.3-70b-versatile` as the agent, Gemini as the LLM
judge):

| Metric | Score |
|---|---|
| Correctness | 1.00 (6/6) |
| Groundedness | 1.00 (6/6) |
| Latency (p50) | ~7.5s per question |

Every answer was both factually correct against ground truth and fully
grounded in the retrieved context — a solid signal that the retrieve/grade/
rewrite loop is doing its job.

## Tuning knobs

Everything's in `src/config.py` — chunk size/overlap, top-k, the similarity
threshold for dropping weak matches, which embedding/LLM model to use, and
how many times the agent is allowed to rewrite a query before giving up.
Defaults are reasonable for general-purpose PDFs but worth tweaking if your
documents are very short/long or very technical.
