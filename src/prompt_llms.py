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
import time

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GOOGLE_AI')

class PromptLLMS:
    def __init__(self, prompt_template, question, question1=None, question2=None, q1=None, q2=None, q3=None):
        self.prompt_template = prompt_template
        self.question = question
        self.question1 = question1
        self.question2 = question2
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

    def execute_on_gemini(self):
        gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=gemini_key,max_tokens=None, temperature=0.1)
        chain = self.prompt_template | gemini

        result =  chain.invoke({"question": self.question})
        return result.content
    
    def execute_on_openAI_model(self,openAI_model):
        gpt_4 = ChatOpenAI(model=openAI_model,openai_api_key=openai_api_key,temperature=0.7)
        chain = self.prompt_template | gpt_4

        result =  chain.invoke({"question": self.question})
        return result.content

    def execute_on_openAI_two_qeustions(self,openAI_model):
        gpt_4 = ChatOpenAI(model=openAI_model,openai_api_key=openai_api_key,temperature=0.7)
        chain = self.prompt_template | gpt_4

        result =  chain.invoke({"q1": self.question1, "q2": self.question2})
        return result.content

    def execute_on_openAI_three_questions(self,openAI_model):
        gpt_4 = ChatOpenAI(model=openAI_model,openai_api_key=openai_api_key,temperature=0.7)
        chain = self.prompt_template | gpt_4

        result =  chain.invoke({"q1": self.q1, "q2": self.q2, "q3": self.q3})
        return result.content
