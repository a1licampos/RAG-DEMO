import random
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
            alexandria_type_learning = state.get("alexandria_type_learning", 0)

            combined_content = "\n".join([doc.page_content for doc in retrieve_docs])


            print("Alexandria Type Learning in NodeChat:", alexandria_type_learning)

            if alexandria_type_learning == 1:
                with open("src/prompts/node_chat_kinestesico.txt", "r") as f:
                    prompt_template = f.read()

            elif alexandria_type_learning == 2:
                with open("src/prompts/node_chat_visual.txt", "r") as f:
                    prompt_template = f.read()
            

            chat_response = self.llmOpenAI._get_chat_response_instructions(
                instructions_message=prompt_template,
                input_message=f"Basado en la siguiente información: {combined_content}, responde a la pregunta: {question}",
            )

            state["chatbot_answer"] = chat_response.response
            state["chatbot_answer_visualization"] = self._select_visualization(chat_response.response)

        except Exception as e:
            logging.error(f"Error in NodeChat run: {e}")
            raise

        return dict(state)
    

    def _select_visualization(self, answer:str):
        try:
            subject_muralla_china_img = [
                "img_muralla_china_1.jpg",
                "img_muralla_china_2.jpg",
                "img_muralla_china_3.jpg",
            ]

            subject_muralla_china_video = [
                "video_muralla_china_1.mp4",
                "video_muralla_china_2.mp4",
            ]

            subject_piramide_maslow_img = [
                "img_piramide_maslow_1.png",
            ]

            subjects_available = [
                "muralla_china",
                "piramide_maslow",
            ]

            with open("src/prompts/node_chat_visualizations.txt", "r") as f:
                prompt_template = f.read()

            chat_response = self.llmOpenAI._get_chat_response_instructions(
                instructions_message=prompt_template,
                input_message=f"Temas disponibles {subjects_available}.\n Respuesta a la pregunta: {answer}.",
            )

            if chat_response.response == "piramide_maslow":
                return subject_piramide_maslow_img[0]
            else:
                roll = random.choice(["img", "video"])
                
                if roll == "img":
                    random_visualization = random.choice(subject_muralla_china_img)
                else:
                    random_visualization = random.choice(subject_muralla_china_video)
                return random_visualization

        except Exception as e:
            logging.error(f"Error in _select_visualization: {e}")
            raise    

#    _____
#   ( \/ @\____
#   /           O
#  /   (_|||||_/
# /____/  |||
#       kimba
