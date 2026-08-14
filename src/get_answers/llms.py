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
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_deepseek import ChatDeepSeek

import logging
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

# Self-hosted (Ollama) settings. All of them are overridable from the environment
# so a run can be pointed at a different host without touching the code.
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://warhol.informatik.rwth-aachen.de:11434')
# Connect timeout guards against an unreachable host, read timeout against a
# server that accepted the request but stopped emitting tokens. Because the
# response is streamed, the read timeout applies *between chunks*, so a slow
# generation never trips it while a genuine stall does.
OLLAMA_CONNECT_TIMEOUT = float(os.getenv('OLLAMA_CONNECT_TIMEOUT', '10'))
OLLAMA_READ_TIMEOUT = float(os.getenv('OLLAMA_READ_TIMEOUT', '600'))
# Keeps the model resident between questions instead of paying a reload every
# time the default 5 minute idle window expires mid-benchmark.
OLLAMA_KEEP_ALIVE = os.getenv('OLLAMA_KEEP_ALIVE', '30m')
# Ollama defaults to a 2048 token context and silently truncates beyond it.
OLLAMA_NUM_CTX = int(os.getenv('OLLAMA_NUM_CTX', '8192'))
OLLAMA_NUM_PREDICT = int(os.getenv('OLLAMA_NUM_PREDICT', '2048'))
OLLAMA_MAX_RETRIES = int(os.getenv('OLLAMA_MAX_RETRIES', '4'))
OLLAMA_RETRY_BACKOFF = float(os.getenv('OLLAMA_RETRY_BACKOFF', '5'))

logger = logging.getLogger(__name__)

# azure_models = ['gpt-5',"gpt-5-mini","gpt-5-nano","gpt-4o","o3"]
azure_models = []
# azure_models = ['gpt-5',"gpt-5-mini","o3","gpt-4o","gpt-4.1",]
openai_models_notemperature = ["gpt-5-mini","o3","gpt-5-nano"]
openai_models = ['gpt-4.1-nano-2025-04-14', 'gpt-4.1-mini-2025-04-14', 'gpt-4.1-2025-04-14',"gpt-4o",'gpt-5']
gemini_models = ["gemini-2.0-flash","gemini-2.5-pro","gemini-2.5-flash"]
xai_models = ['grok-3-mini','grok-4-0709']
claude_models = ['claude-3-5-sonnet-20240620','claude-3-haiku']
self_hosted_models = ['llama3.1:8b','llama3.1:70b', 'deepseek-r1:1.5b', 'deepseek-r1:70b', 'gpt-oss:20b', 'mistral-small:24b']
deepseek_models = ['deepseek-chat','deepseek-reasoner']

class SelfHostedAPIError(RuntimeError):
    """Raised when the self-hosted endpoint fails after all retries."""


def _role_for(message: BaseMessage) -> str:
    """Map a LangChain message onto the role names Ollama's chat API expects."""
    return {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }.get(message.type, "user")


class SelfHostedAPIWrapper(BaseChatModel):
    """Chat model backed by a self-hosted Ollama instance.

    Uses ``/api/chat`` rather than ``/api/generate`` so the server applies the
    model's own chat template to the system/user turns, matching how the hosted
    provider APIs treat the same prompts.
    """

    model: str
    base_url: str = OLLAMA_BASE_URL
    temperature: float = 0.0
    num_ctx: int = OLLAMA_NUM_CTX
    num_predict: int = OLLAMA_NUM_PREDICT
    keep_alive: str = OLLAMA_KEEP_ALIVE
    connect_timeout: float = OLLAMA_CONNECT_TIMEOUT
    read_timeout: float = OLLAMA_READ_TIMEOUT
    max_retries: int = OLLAMA_MAX_RETRIES
    retry_backoff: float = OLLAMA_RETRY_BACKOFF

    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "self-hosted-ollama-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": _role_for(m), "content": m.text()} for m in messages
            ],
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        if stop:
            payload["options"]["stop"] = list(stop)

        content = self._post_with_retries(payload)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _post_with_retries(self, payload: dict) -> str:
        """POST to the chat endpoint, retrying transient failures with backoff."""
        url = f"{self.base_url.rstrip('/')}/api/chat"
        headers = {"Content-Type": "application/json"}
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self._post_once(url, payload, headers)
            except SelfHostedAPIError:
                # Permanent failure (4xx, or an error reported by the server).
                raise
            except (requests.RequestException, json.JSONDecodeError) as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    break
                delay = self.retry_backoff * (2 ** attempt)
                logger.warning(
                    "Self-hosted request to %s failed (attempt %s/%s): %s. "
                    "Retrying in %.0fs.",
                    url, attempt + 1, self.max_retries, e, delay,
                )
                time.sleep(delay)

        raise SelfHostedAPIError(
            f"Self-hosted request to {url} failed after {self.max_retries} "
            f"attempts: {last_error}"
        ) from last_error

    def _post_once(self, url: str, payload: dict, headers: dict) -> str:
        response = requests.post(
            url,
            data=json.dumps(payload),
            headers=headers,
            stream=True,
            timeout=(self.connect_timeout, self.read_timeout),
        )
        try:
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise SelfHostedAPIError(
                    f"{url} returned {response.status_code}: {response.text[:500]}"
                )
            response.raise_for_status()

            aggregated_response = ""
            for line in response.iter_lines():
                if not line:
                    continue
                obj = json.loads(line.decode("utf-8"))
                if "error" in obj:
                    raise SelfHostedAPIError(f"{url} reported: {obj['error']}")
                chunk = obj.get("message", {}).get("content")
                if chunk:
                    aggregated_response += chunk
            return aggregated_response
        finally:
            # Release the socket even when the stream is abandoned part-way.
            response.close()

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
    elif model_name in openai_models_notemperature:
        return ChatOpenAI(model=model_name, openai_api_key=openai_api_key)
    elif model_name in openai_models:
        return ChatOpenAI(model=model_name, openai_api_key=openai_api_key,temperature=temperature)
    elif model_name in gemini_models:
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key, max_tokens=max_tokens, temperature=temperature)
    elif model_name in xai_models:
        return ChatXAI(model=model_name, xai_api_key=XAI_API_KEY, max_tokens=max_tokens, temperature=temperature)
    elif model_name in claude_models:
        return ChatAnthropic(model=model_name, anthropic_api_key=ANTHROPIC_API_KEY, temperature=temperature)
    elif model_name in self_hosted_models:
        return SelfHostedAPIWrapper(model=model_name, temperature=temperature)
    elif model_name in deepseek_models:
        return ChatDeepSeek(model=model_name,temperature=temperature,max_tokens=None)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
