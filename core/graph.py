import subprocess
import tempfile
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


class State(TypedDict):
    graph_state: str


def node_1(state):
    print("---Node 1---")
    return {"graph_state": state["graph_state"] + "Я"}


def node_2(state):
    print("---Node 2---")
    return {"graph_state": state["graph_state"] + "люблю"}


def node_3(state):
    print("---Node 3---")
    return {"graph_state": state["graph_state"] + "Russia!"}


builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)


graph = builder.compile()

img = Image(graph.get_graph().draw_mermaid_png())

with tempfile.NamedTemporaryFile() as f:
    f.write(img.data)  # type: ignore
    subprocess.run(["xdg-open", f.name], check=True)
