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

import time

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GOOGLE_API_KEY')
XAI_API_KEY = os.getenv('XAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_api_version= os.getenv('AZURE_API_VERSION', '2024-12-01-preview')

azure_models = ["gpt-4o","o3","o1"]
openai_models = ['gpt-4.1-nano-2025-04-14', 'gpt-4.1-mini-2025-04-14', 'gpt-4.1-2025-04-14']
gemini_models = ["gemini-2.0-flash","gemini-2.5-pro"]
xai_models = ['grok-3-mini','grok-4-0709']
claude_models = ['claude-3-5-sonnet-20240620','claude-3-haiku']




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
        return result.content

    def execute_two_question(self):
        chain = self.prompt_template | self.chat

        result =  chain.invoke({"q1": self.question1, "q2": self.question2})
        return result.content

    def execute_three_question(self):
        chain = self.prompt_template | self.chat

        result =  chain.invoke({"q1": self.q1, "q2": self.q2, "q3": self.q3})
        return result.content


def return_chat_model(model_name, temperature=0):
    if model_name in openai_models:
        return ChatOpenAI(model=model_name, openai_api_key=openai_api_key, temperature=temperature)
    elif model_name in gemini_models:
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key, max_tokens=None, temperature=temperature)
    elif model_name in azure_models:
        return AzureChatOpenAI(azure_deployment=model_name, api_version=azure_api_version,)
    elif model_name in xai_models:
        return ChatXAI(model=model_name, xai_api_key=XAI_API_KEY, temperature=temperature)
    elif model_name in claude_models:
        return ChatAnthropic(model=model_name, anthropic_api_key=ANTHROPIC_API_KEY, temperature=temperature)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
