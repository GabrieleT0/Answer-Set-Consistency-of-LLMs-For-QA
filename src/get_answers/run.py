from .single_question_benchmark import run_benchmark_equal
from .relation_classification_and_questions import run_relation_classification_and_questions
from .try_fix_llm_response import run_fix_llm_response


def main():
    print("Starting unified LLM benchmark pipeline...\n")

    print("\nStep 1: Running single question benchmark")
    run_benchmark_equal()

    print("\nStep 2: Trying to fix LLM responses")
    run_fix_llm_response()

    print("Step 3: Running relation classification and question generation")
    run_relation_classification_and_questions()


    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()
