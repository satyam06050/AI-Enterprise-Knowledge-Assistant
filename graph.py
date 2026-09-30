from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from agents import manager_agent, generate_answer, generate_uploaded_answer
from rag import retrieve

UPLOADED_NOT_FOUND_MESSAGE = "I could not find this information in the uploaded document."

class AgentState(TypedDict):
    question: str
    category: str
    answer: str
    context: str
    sources: list


def manager_node(state, chunks, metadata, index):
    question = state["question"]
    category = manager_agent(question)

    department_map = {
        "HR": "hr",
        "TECHNICAL": "technical",
        "PROJECT": "projects",
        "GENERAL": None
    }

    results = retrieve(
        question, chunks, metadata, index,
        department=department_map[category],
        top_k=5
    )

    context_parts = []
    sources = []

    for result in results:
        m = result["metadata"]
        context_parts.append(
            f"\nSource:\n{m['source']}\n\nPage:\n{m['page']}\n\nDepartment:\n{m['department']}\n\nContent:\n{result['text']}\n"
        )
        sources.append(m)

    return {
        "category": category,
        "context": "\n".join(context_parts),
        "sources": sources
    }


def answer_node(state):
    agent_name_map = {
        "HR": "HR specialist agent",
        "TECHNICAL": "Technical specialist agent",
        "PROJECT": "Project specialist agent",
        "GENERAL": "General company knowledge agent"
    }
    answer = generate_answer(
        state["question"],
        state["context"],
        agent_name_map[state["category"]]
    )
    return {"answer": answer}


def create_graph(chunks, metadata, index):
    graph = StateGraph(AgentState)

    def manager_wrapper(state):
        return manager_node(state, chunks, metadata, index)

    graph.add_node("manager", manager_wrapper)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "manager")
    graph.add_edge("manager", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def uploaded_manager_node(state, chunks, metadata, index, document_id):
    results = retrieve(
        state["question"],
        chunks,
        metadata,
        index,
        document_id=document_id,
        top_k=5,
    )

    context_parts = []
    sources = []
    for result in results:
        source = result["metadata"]
        location = f"Page {source['page']}" if source.get("page") else f"Row {source['row']}" if source.get("row") else "Document"
        context_parts.append(
            f"\nSource: {source['source']}\nLocation: {location}\nContent: {result['text']}\n"
        )
        sources.append(source)

    return {"context": "\n".join(context_parts), "sources": sources}


def uploaded_answer_node(state):
    if not state["context"].strip():
        return {"answer": UPLOADED_NOT_FOUND_MESSAGE}

    answer = generate_uploaded_answer(state["question"], state["context"])
    if UPLOADED_NOT_FOUND_MESSAGE.lower() in answer.lower():
        return {"answer": UPLOADED_NOT_FOUND_MESSAGE}
    return {"answer": answer}


def create_uploaded_graph(chunks, metadata, index, document_id):
    """Build a LangGraph Q&A path constrained to one session-uploaded document."""
    graph = StateGraph(AgentState)

    def manager_wrapper(state):
        return uploaded_manager_node(state, chunks, metadata, index, document_id)

    graph.add_node("retrieve_uploaded", manager_wrapper)
    graph.add_node("answer_uploaded", uploaded_answer_node)
    graph.add_edge(START, "retrieve_uploaded")
    graph.add_edge("retrieve_uploaded", "answer_uploaded")
    graph.add_edge("answer_uploaded", END)

    return graph.compile()
