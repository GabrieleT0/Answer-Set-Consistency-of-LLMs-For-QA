import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
import time

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GOOGLE_AI')

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


def return_chat_model(model_name, temperature=0.0):
    if 'gpt' in model_name:
        return ChatOpenAI(model=model_name, openai_api_key=openai_api_key, temperature=temperature)
    elif 'gemini' in model_name:
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key, max_tokens=None, temperature=temperature)
    elif 'deepseek' in model_name:
        return ChatDeepSeek(model=model_name,temperature=temperature,max_tokens=None)
    else:
        raise ValueError(f"Model {model_name} is not supported.")