import os
import csv
import json
from langchain_core.prompts import ChatPromptTemplate
try:
    from .llms import PromptLLMS
    from . import utils
except ImportError:
    from llms import PromptLLMS
    import utils
import yaml
import datetime
import logging

LOGICAL_RELATIONS_MAP = {
                'Q1': 'equal', 'Q2': 'equal', 'Q3': 'sup-sub', 'Q4': 'minus'
            }

# Conditional logging
def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.datetime.now().strftime("single_question_benchmark_%Y-%m-%d_%H-%M.log")
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding='utf-8')
        ]
    )
    for name in logging.root.manager.loggerDict:
        if name not in ["single_question_benchmark"]:  # your custom logger name
            logging.getLogger(name).setLevel(logging.WARNING)

    logger = logging.getLogger("single_question_benchmark")
    return logger
# Load environment variables

root_dir = os.path.dirname(os.path.abspath(__name__))

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(HERE, "prompts.yaml")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)


# === Modules ===

def load_questions(tsv_file, column):
    questions = []
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            questions.append(row[column])
    return questions

def get_prompt(prompt_type, language):
    template = PROMPTS[prompt_type][language]
    instructions = template.replace("{question}", "").strip()
    return ChatPromptTemplate.from_messages(
        [
            ("system", instructions),
            ("human", "{question}"),
        ]
    )


def _is_content_filter_error(error):
    """Return whether an exception represents an Azure content-filter block."""
    payload = getattr(error, "body", None)
    error_text = str(error)
    if payload is not None:
        try:
            error_text += " " + json.dumps(payload)
        except TypeError:
            error_text += " " + str(payload)
    error_text = error_text.lower()
    return (
        "content_filter" in error_text
        or "responsibleaipolicyviolation" in error_text
        or "content filter being triggered" in error_text
    )


def process_question(question, llm_model, prompt_template, language, logger):
    try:
        llms = PromptLLMS(model=llm_model, prompt_template=prompt_template, question=question)
        response = llms.execute_single_question()
        if language == 'en':
            return utils.convert_response_to_set(response)
        else:
            return utils.convert_response_to_set_es(response)
    except Exception as e:
        if not _is_content_filter_error(e):
            raise
        logger.info(f"Content filter triggered for question: {question}")
        logger.info(f"Error: {e}")
        return None  # fallback response set


def _answer_path(
    config,
    dataset,
    column,
    language,
    prompt_type,
    llm_model,
    *,
    legacy_model_name=False,
):
    """Return an output path that keeps prompt ablations isolated."""
    lang_prefix = '' if language == 'en' else '*'
    relation = LOGICAL_RELATIONS_MAP[column]
    action_dir = "zero-shot-no-idk" if prompt_type == "standard_no_idk" else "zero-shot"
    prompt_prefix = "wikidata_" if prompt_type == "wikidata" else ""
    output_model = (
        llm_model
        if legacy_model_name
        else utils.output_model_name(llm_model)
    )
    suffix = f"_answers_{prompt_prefix}{output_model}.json"
    project_root = config.get("root_dir", root_dir)
    return os.path.join(
        project_root,
        "data",
        "answers",
        action_dir,
        dataset.split(".")[0],
        relation,
        f"{lang_prefix}{column}_{relation}{suffix}",
    )


def _content_filter_path(
    config, dataset, column, language, prompt_type, llm_model
):
    answer_path = _answer_path(
        config, dataset, column, language, prompt_type, llm_model
    )
    stem, extension = os.path.splitext(answer_path)
    return f"{stem}_content_filters{extension}"


def load_content_filters(
    config, dataset, column, language, prompt_type, llm_model
):
    path = _content_filter_path(
        config, dataset, column, language, prompt_type, llm_model
    )
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    legacy_answer_path = _answer_path(
        config,
        dataset,
        column,
        language,
        prompt_type,
        llm_model,
        legacy_model_name=True,
    )
    stem, extension = os.path.splitext(legacy_answer_path)
    legacy_path = f"{stem}_content_filters{extension}"
    if legacy_path != path and os.path.exists(legacy_path):
        with open(legacy_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_content_filters(
    filtered_questions,
    config,
    dataset,
    column,
    language,
    prompt_type,
    llm_model,
):
    path = _content_filter_path(
        config, dataset, column, language, prompt_type, llm_model
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            filtered_questions, f, ensure_ascii=False, indent=4
        )


def save_answers(answers, dataset, column, language, prompt_type, llm_model, config):
    out_path = _answer_path(
        config, dataset, column, language, prompt_type, llm_model
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
    # logger.info(f"Answers saved to {out_path}")


def load_answers(dataset, column, language, prompt_type, llm_model, config):
    in_file = _answer_path(
        config, dataset, column, language, prompt_type, llm_model
    )

    if os.path.exists(in_file):
        with open(in_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    legacy_file = _answer_path(
        config,
        dataset,
        column,
        language,
        prompt_type,
        llm_model,
        legacy_model_name=True,
    )
    if legacy_file != in_file and os.path.exists(legacy_file):
        with open(legacy_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}



# === benchmark ===

def run_benchmark_equal(prompt_type, config, logger):
    project_root = config.get("root_dir", root_dir)
    for language in config["languages"]:
        for llm_model in config["llm_models"]:
            for dataset in config["datasets"]:
                logger.info(f"Processing dataset: {dataset} for model: {llm_model} and language: {language}")
                tsv_file = os.path.join(
                    project_root, "data", "ASCB", language, dataset
                )

                for column in ['Q1', 'Q2', 'Q3', 'Q4']:
                    logger.info(f"Processing column: {column}")
                    questions = load_questions(tsv_file, column)
                    prompt_template = get_prompt(prompt_type, language)

                    answers = load_answers(dataset, column, language, prompt_type, llm_model, config)
                    filtered_questions = load_content_filters(
                        config,
                        dataset,
                        column,
                        language,
                        prompt_type,
                        llm_model,
                    )
                    retry_filtered = config.get(
                        "retry_content_filtered", False
                    )

                    for index, question in enumerate(questions):
                        # An empty list is a valid "no answer" result, so key
                        # presence—not answer length—determines resumability.
                        if str(index) in answers:
                            continue
                        if (
                            str(index) in filtered_questions
                            and not retry_filtered
                        ):
                            logger.info(
                                "Question %s was previously content-filtered; "
                                "skipping its API call.",
                                index + 1,
                            )
                            continue

                        response_set = process_question(question, llm_model, prompt_template, language, logger)
                        if response_set is None:
                            filtered_questions[str(index)] = {
                                "question": question,
                                "reason": "content_filter",
                            }
                            save_content_filters(
                                filtered_questions,
                                config,
                                dataset,
                                column,
                                language,
                                prompt_type,
                                llm_model,
                            )
                            logger.info(
                                "Checkpointed question %s as content-filtered; "
                                "continuing the benchmark.",
                                index + 1,
                            )
                            continue

                        answers[str(index)] = response_set
                        if str(index) in filtered_questions:
                            del filtered_questions[str(index)]
                            save_content_filters(
                                filtered_questions,
                                config,
                                dataset,
                                column,
                                language,
                                prompt_type,
                                llm_model,
                            )

                        logger.info(f"Question {index + 1}: {question}")
                        # logger.info(f"LLM Response: {response_set}")

                        save_answers(answers, dataset, column, language, prompt_type, llm_model, config)

                    save_answers(answers, dataset, column, language, prompt_type, llm_model, config)


def main(config = None, logger = setup_logger()):
    if config == None:
        config = {
            "languages": ['en'],
            "llm_models": ['o3'],
            "datasets": ['spinach.tsv', 'qawiki.tsv', 'synthetic.tsv'],
            "prompt_types": ['standard', 'wikidata']
        }
    

    for prompt_type in config["prompt_types"]:
        run_benchmark_equal(prompt_type, config, logger)

if __name__ == "__main__":
    main()
