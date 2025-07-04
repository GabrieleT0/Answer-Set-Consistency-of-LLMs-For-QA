import os
import requests
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from abc import ABC, abstractmethod

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GOOGLE_AI')

class PromptLLM(ABC):
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    @abstractmethod
    def query(self, **kwargs) -> str:
        """
        Query the LLM with the given variables.
        kwargs: variables for the prompt template (e.g., question, q1, q2, q3)
        """
        return "idk"

class GoogleGenerativeAILLM(PromptLLM):
    def __init__(self, prompt_template, model="gemini-2.5-pro"):
        super().__init__(prompt_template)
        self.model = model
        self.api_key = gemini_key

    def query(self, **kwargs) -> str:
        gemini = ChatGoogleGenerativeAI(model=self.model, google_api_key=self.api_key, max_tokens=None, temperature=0.1)
        chain = self.prompt_template | gemini
        result = chain.invoke(kwargs)
        return result.content

class OpenAILLM(PromptLLM):
    def __init__(self, prompt_template, model="gpt-4"):
        super().__init__(prompt_template)
        self.model = model
        self.api_key = openai_api_key

    def query(self, **kwargs) -> str:
        gpt = ChatOpenAI(model=self.model, openai_api_key=self.api_key, temperature=0.7)
        chain = self.prompt_template | gpt
        result = chain.invoke(kwargs)
        return result.content

class CustomServerLLM(PromptLLM):
    def __init__(self, prompt_template, model="llama2:latest", url="http://warhol.informatik.rwth-aachen.de:11434/api/generate"):
        super().__init__(prompt_template)
        self.model = model
        self.url = url

    def query(self, **kwargs) -> str:
        prompt = self.prompt_template.format(**kwargs)
        payload = {
            "model": self.model,
            "prompt": prompt
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.url, data=json.dumps(payload), headers=headers, stream=True)
        response.raise_for_status()
        aggregated_response = ""
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        obj = json.loads(line.decode("utf-8"))
                        if "response" in obj:
                            aggregated_response += obj["response"]
                    except Exception:
                        continue
            return aggregated_response
        except Exception:
            return response.text
