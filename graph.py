from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from agents import manager_agent, generate_answer
from rag import retrieve


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
