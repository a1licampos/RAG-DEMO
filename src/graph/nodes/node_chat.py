import logging
from src.graph.state.graph_state import State
from src.core.llm.llm_openai import LLMOpenAI

class NodeChat:

    def __init__(self):
        self.llmOpenAI = LLMOpenAI()

    def run(self, state:State):
        try:
            question = state.get("user_question", "")
            retrieve_docs = state.get("node_retrieve_docs", [])

            combined_content = "\n".join([doc.page_content for doc in retrieve_docs])

            with open("src/prompts/node_chat.txt", "r") as f:
                prompt_template = f.read()

            chat_response = self.llmOpenAI._get_chat_response_instructions(
                instructions_message=prompt_template,
                input_message=f"Basado en la siguiente información: {combined_content}, responde a la pregunta: {question}",
            )

            state["chatbot_answer"] = chat_response.response

        except Exception as e:
            logging.error(f"Error in NodeChat run: {e}")
            raise

        return dict(state)

#    _____
#   ( \/ @\____
#   /           O
#  /   (_|||||_/
# /____/  |||
#       kimba
