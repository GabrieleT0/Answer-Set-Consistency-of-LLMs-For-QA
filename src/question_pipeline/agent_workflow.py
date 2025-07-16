import re
import json
import yaml
from typing import Optional
from openai import AzureOpenAI


# ========================== #
#         CONFIG             #
# ========================== #
# Load config
with open("config.json", "r") as f:
    CONFIG = json.load(f)

DEBUG = CONFIG["debug"]
MODEL_NAME = CONFIG["model_name"]
AZURE_API_VERSION = CONFIG["api_version"]
PROMPT_PATH = CONFIG["prompt_path"]
INPUT_PATH = CONFIG["input_path"]
OUTPUT_PATH = CONFIG["output_path"]
API_KEY = CONFIG["OPENAI_API_KEY"]
ENDPOINT_URL = CONFIG["ENDPOINT_URL"]



# ========================== #
#        INIT CLIENT         #
# ========================== #
# Azure client init
client = AzureOpenAI(
    azure_endpoint=ENDPOINT_URL,
    api_key=API_KEY,
    api_version=AZURE_API_VERSION,
)

with open(PROMPT_PATH, "r") as f:
    PROMPTS = yaml.safe_load(f)

# ========================== #
#      CORE FUNCTIONS        #
# ========================== #
def call_llm(prompt: str, model: str = MODEL_NAME, temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    result = response.choices[0].message.content.strip()
    if DEBUG:
        print(f"LLM response:\n{result}\n")
    return result

def format_prompt(agent_key: str, lang: str = "en", **kwargs) -> str:
    template = PROMPTS[agent_key][lang]
    return template.format(**kwargs)

# ========================== #
#        AGENT LOGIC         #
# ========================== #
def agent1_validate_q1(q1: str) -> bool:
    prompt = format_prompt("Agent1_FilterQ1", question=q1)
    result = call_llm(prompt)
    return result.lower().startswith("yes")

def agent2_generate_q2(q1: str) -> str:
    prompt = format_prompt("Agent2_GenerateQ2", question=q1)
    response = call_llm(prompt)
    match = re.search(r"(?i)^q2:\s*(.+)", response, re.MULTILINE)
    return match.group(1).strip() if match else response.strip()

def agent3_generate_q3_q4(q1: str) -> tuple[str, str]:
    prompt = format_prompt("Agent3_GenerateQ3Q4", question=q1)
    response = call_llm(prompt)
    q3 = re.search(r"(?i)^q3:\s*(.+)", response, re.MULTILINE)
    q4 = re.search(r"(?i)^q4:\s*(.+)", response, re.MULTILINE)
    return (
        q3.group(1).strip() if q3 else "",
        q4.group(1).strip() if q4 else ""
    )

def agent4_validate_all(q1: str, q2: str, q3: str, q4: str) -> tuple[bool, str, str, str, str]:
    prompt = format_prompt("Agent4_ValidateAll", q1=q1, q2=q2, q3=q3, q4=q4)
    result = call_llm(prompt)

    if result.strip().lower() == "yes":
        return True, q1, q2, q3, q4

    def extract(field: str, fallback: str) -> str:
        match = re.search(fr"(?i)^corrected {field}:\s*(.+)", result, re.MULTILINE)
        return match.group(1).strip() if match else fallback

    return False, extract("q1", q1), extract("q2", q2), extract("q3", q3), extract("q4", q4)

def run_agent_workflow(q1_input: str) -> Optional[dict]:
    if not agent1_validate_q1(q1_input):
        return None

    q2 = agent2_generate_q2(q1_input)
    q3, q4 = agent3_generate_q3_q4(q1_input)
    if not q3 or not q4:
        return None

    valid, q1_final, q2_final, q3_final, q4_final = agent4_validate_all(q1_input, q2, q3, q4)

    return {
        "Q1": q1_final,
        "Q2": q2_final,
        "Q3": q3_final,
        "Q4": q4_final
    }
