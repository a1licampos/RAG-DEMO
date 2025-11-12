from src.graph.state.graph_state import State, TaskRoute
from src.core.llm.llm_openai import LLMOpenAI


class NodeRouter:

    def __init__(self):
        self.llmOpenAI = LLMOpenAI()

    def run(self, state:State):
        try:
            chat_response = self.llmOpenAI._get_chat_response(
                system_message="Eres un descriminador de preguntas.",
                developer_message="Retornar true si la pregunta es educativa, false en caso contrario.",
                user_message=f"Responde a la siguiente pregunta: {state['user_question']}",
                text_format=TaskRoute
            )
            return chat_response
        except Exception as e:
            print(f"Error in NodeRouter run: {e}")
            raise