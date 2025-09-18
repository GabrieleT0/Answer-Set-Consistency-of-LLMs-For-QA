import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_vertexai import ChatVertexAI
from langchain_xai import ChatXAI
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
# from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models.llms import LLM
from langchain_deepseek import ChatDeepSeek

import time
import requests
import json

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GOOGLE_API_KEY')
XAI_API_KEY = os.getenv('XAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_api_version= os.getenv('AZURE_API_VERSION', '2024-12-01-preview')

azure_models = ['gpt-5',"gpt-5-mini","gpt-5-nano","gpt-4o","o3"]
# azure_models = ['gpt-5',"gpt-5-mini","o3","gpt-4o","gpt-4.1",]
openai_models_notemperature = ["o3","gpt-5-nano"]
openai_models = ['gpt-5',"gpt-5-mini","gpt-5-nano",'gpt-4.1-nano-2025-04-14', 'gpt-4.1-mini-2025-04-14', 'gpt-4.1-2025-04-14',"gpt-4o","gpt-4.1"]
gemini_models = ["gemini-2.0-flash","gemini-2.5-pro","gemini-2.5-flash"]
xai_models = ['grok-3-mini','grok-4-0709']
claude_models = ['claude-3-5-sonnet-20240620','claude-3-haiku']
self_hosted_models = ['llama3.1:8b','llama3.1:70b', 'deepseek-r1:1.5b', 'deepseek-r1:70b', 'gpt-oss:20b']
deepseek_models = ['deepseek-chat','deepseek-reasoner']

class SelfHostedAPIWrapper(LLM):
    model: str
    url: str = 'your_api_url'

    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters."""
        return {"model": self.model, "url": self.url}

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "custom"

    def _call(self, prompt: str, stop = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt
        }
        headers = {"Content-Type": "application/json"}
    
        response = requests.post(self.url, data=json.dumps(payload), headers=headers, stream=True)
        response.raise_for_status()

        aggregated_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    obj = json.loads(line.decode("utf-8"))
                    if "response" in obj:
                        aggregated_response += obj["response"]
                except Exception:
                    continue
        return aggregated_response

class PromptLLMS:
    def __init__(self, model, prompt_template, question=None, question1=None, question2=None, q1=None, q2=None, q3=None):
        self.chat = return_chat_model(model)
        self.prompt_template = prompt_template
        self.question = question
        self.question1 = question1
        self.question2 = question2
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

    def execute_single_question(self):
        chain = self.prompt_template | self.chat

        result =  chain.invoke({"question": self.question})
        return result

    def execute_two_question(self):
        chain = self.prompt_template | self.chat

        result =  chain.invoke({"q1": self.question1, "q2": self.question2})
        return result

    def execute_three_question(self):
        chain = self.prompt_template | self.chat

        result =  chain.invoke({"q1": self.q1, "q2": self.q2, "q3": self.q3})
        return result


def return_chat_model(model_name, temperature=0, max_tokens = 20000):
    if model_name in azure_models:
        return AzureChatOpenAI(azure_deployment=model_name, api_version=azure_api_version,)
    elif model_name in openai_models:
        return ChatOpenAI(model=model_name, openai_api_key=openai_api_key)
    elif model_name in openai_models_notemperature:
        return ChatOpenAI(model=model_name, openai_api_key=openai_api_key, max_tokens=max_tokens)
    elif model_name in gemini_models:
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key, max_tokens=max_tokens, temperature=temperature)
    elif model_name in xai_models:
        return ChatXAI(model=model_name, xai_api_key=XAI_API_KEY, max_tokens=max_tokens, temperature=temperature)
    elif model_name in claude_models:
        return ChatAnthropic(model=model_name, anthropic_api_key=ANTHROPIC_API_KEY, temperature=temperature)
    elif model_name in self_hosted_models:
        return SelfHostedAPIWrapper(model=model_name, url="http://warhol.informatik.rwth-aachen.de:11434/api/generate")
    elif model_name in deepseek_models:
        return ChatDeepSeek(model=model_name,temperature=temperature,max_tokens=None)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
