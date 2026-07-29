"""Agentic RAG: retrieve -> grade -> (rewrite + retry | generate).

Unlike a plain RAG chain, this graph checks whether what it retrieved from the
real PDF corpus actually addresses the question before answering. If not, it
rewrites the search query and tries again (bounded by MAX_QUERY_REWRITES)
before falling back to an ungrounded answer.
"""
from typing import Any, Dict, List, Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from .config import GROQ_API_KEY, LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE, MAX_QUERY_REWRITES
from .embeddings import EmbeddingManager
from .retriever import RAGRetriever
from .vectorstore import VectorStore


class AgentState(TypedDict):
    question: str
    search_query: str
    documents: List[Dict[str, Any]]
    grounded: bool
    rewrites: int
    answer: str


_llm: Optional[ChatGroq] = None
_retriever: Optional[RAGRetriever] = None
_graph = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _llm


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(VectorStore(), EmbeddingManager())
    return _retriever


def retrieve_documents(state: AgentState) -> AgentState:
    hits = get_retriever().retrieve(state["search_query"])
    return {**state, "documents": hits}


def grade_documents(state: AgentState) -> AgentState:
    """Ask the LLM whether the retrieved chunks actually address the question."""
    if not state["documents"]:
        return {**state, "grounded": False}

    context = "\n\n".join(doc["content"] for doc in state["documents"])
    prompt = (
        "You are grading whether a CONTEXT is relevant enough to answer a QUESTION.\n"
        f"QUESTION: {state['question']}\n\nCONTEXT:\n{context}\n\n"
        "Reply with exactly one word: RELEVANT or NOT_RELEVANT."
    )
    verdict = get_llm().invoke(prompt).content.strip().upper()
    return {**state, "grounded": verdict.startswith("RELEVANT")}


def rewrite_query(state: AgentState) -> AgentState:
    """Reformulate the search query when the first retrieval missed the mark."""
    prompt = (
        f'The search query "{state["search_query"]}" did not retrieve documents relevant to the '
        f'question "{state["question"]}". Rewrite it as a single, more specific search query. '
        "Reply with only the rewritten query."
    )
    new_query = get_llm().invoke(prompt).content.strip()
    return {**state, "search_query": new_query, "rewrites": state["rewrites"] + 1}


def generate_answer(state: AgentState) -> AgentState:
    if state["grounded"]:
        context = "\n\n".join(doc["content"] for doc in state["documents"])
        prompt = (
            f"Use the following context to answer the question concisely.\n\nContext:\n{context}\n\n"
            f"Question: {state['question']}\n\nAnswer:"
        )
    else:
        prompt = (
            "No relevant context was found in the knowledge base for this question. "
            "Answer from general knowledge if you can, and say so explicitly.\n\n"
            f"Question: {state['question']}"
        )
    response = get_llm().invoke(prompt)
    return {**state, "answer": response.content}


def route_after_grading(state: AgentState) -> str:
    if state["grounded"]:
        return "generate"
    if state["rewrites"] < MAX_QUERY_REWRITES:
        return "rewrite"
    return "generate"


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade", route_after_grading, {"generate": "generate", "rewrite": "rewrite"}
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    return workflow.compile()


def ask_question(question: str) -> Dict[str, Any]:
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial_state: AgentState = {
        "question": question,
        "search_query": question,
        "documents": [],
        "grounded": False,
        "rewrites": 0,
        "answer": "",
    }
    return _graph.invoke(initial_state)
