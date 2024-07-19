import os
from openai import OpenAI

from typing import List, Dict
from transformers.agents.llm_engine import MessageRole, get_clean_message_list
from huggingface_hub import InferenceClient

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class OpenAIEngine:
    def __init__(self, model_name="gpt-4o-mini", temperature=0.5):
        assert model_name
        assert temperature
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
        )
        self.temperature = temperature

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
