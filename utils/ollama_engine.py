import os
from transformers.agents.llm_engine import MessageRole, get_clean_message_list
import ollama

ollama_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class OllamaEngine:
    def __init__(self, model_name, temperature):
        assert model_name
        assert temperature
        self.model_name = model_name
        self.client = ollama.Client()
        self.temperature = temperature

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=ollama_role_conversions)

        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={
                'stop': stop_sequences,
                'temperature': self.temperature,
            },
        )
        return response['message']['content']

# from ollama import Client
# client = Client(host='http://localhost:11434')
# response = client.chat(model='llama3', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])

# import ollama
# response = ollama.chat(model='llama3', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])
