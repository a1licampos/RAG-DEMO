from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.graph.state.graph_state import State


class GraphBuilder:

    def __init__(self):
        pass

    def build_graph(self):
        try:
            builder = StateGraph(State)

        except Exception as e:
            print(f"Error building graph: {e}")
            raise