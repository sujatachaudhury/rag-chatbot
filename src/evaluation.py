"""LangSmith evaluation of the agent: correctness against ground truth, and
groundedness against the documents it actually retrieved.
"""
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client
from typing_extensions import Annotated, TypedDict

from .agent import ask_question

DATASET_NAME = "YTRAG Agent Evaluation"

# Question/ground-truth pairs drawn from Denoising_Report.pdf in data/pdfs/.
# Add more as the corpus grows.
EXAMPLES: List[Dict[str, Dict[str, str]]] = [
    {
        "inputs": {
            "question": "Which denoising method achieved the best overall PSNR and SSIM in the report, and what were the values?"
        },
        "outputs": {
            "answer": "The pretrained Neighbor2Neighbor variant achieved the best overall performance, at 30.56 dB PSNR and 0.883 SSIM."
        },
    },
    {
        "inputs": {
            "question": "Among the single-image denoising methods (no pretraining), which had the highest PSNR and which had the highest SSIM?"
        },
        "outputs": {
            "answer": "Deep Image Prior (DIP) had the highest PSNR among single-image methods at 28.44 dB, while Self2Self had the highest SSIM at 0.821."
        },
    },
    {
        "inputs": {
            "question": "What noise model and noise level was used to corrupt the CBSD68 images, and what was the resulting average noisy PSNR?"
        },
        "outputs": {
            "answer": "Additive white Gaussian noise (AWGN) with standard deviation sigma = 25, giving an average noisy PSNR of about 20.59 dB."
        },
    },
    {
        "inputs": {
            "question": "How does Neighbor2Neighbor construct its self-supervised training pairs?"
        },
        "outputs": {
            "answer": "It samples spatially disjoint sub-images from each 2x2 pixel neighborhood: two diagonally opposite pixels form a training pair, and the other two form a regularization pair, producing an implicit Noise2Noise training objective without needing extra noisy observations."
        },
    },
    {
        "inputs": {
            "question": "Which single-image method was fastest, and how does its inference time compare to Deep Image Prior?"
        },
        "outputs": {
            "answer": "Self2Self was the fastest single-image method at an average of 1751.5 seconds per image, roughly half the time of DIP's 3305.8 seconds."
        },
    },
    {
        "inputs": {
            "question": "What hardware was used to run the denoising experiments?"
        },
        "outputs": {
            "answer": "An Intel Core i5-9300H CPU (4 cores, 8 threads, 2.40 GHz), 8 GB of system RAM, and an NVIDIA GeForce GTX 1650 GPU with 4 GB of VRAM."
        },
    },
]


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the grade"]
    correct: Annotated[bool, ..., "True if the answer matches the ground truth"]


class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the grade"]
    grounded: Annotated[bool, ..., "True if the answer is supported by the retrieved documents"]


CORRECTNESS_INSTRUCTIONS = (
    "You are grading a student's answer against a ground truth answer. Grade ONLY on factual "
    "accuracy relative to the ground truth; extra correct detail is fine."
)
GROUNDED_INSTRUCTIONS = (
    "You are checking whether a STUDENT ANSWER is grounded in the given FACTS, with no hallucinated "
    "claims outside those facts."
)


def _grade(schema, instructions: str, user_content: str) -> Dict[str, Any]:
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0).with_structured_output(
        schema, method="json_schema", strict=True
    )
    return llm.invoke([{"role": "system", "content": instructions}, {"role": "user", "content": user_content}])


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    content = (
        f"QUESTION: {inputs['question']}\nGROUND TRUTH: {reference_outputs['answer']}\n"
        f"STUDENT ANSWER: {outputs['answer']}"
    )
    return _grade(CorrectnessGrade, CORRECTNESS_INSTRUCTIONS, content)["correct"]


def groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    facts = "\n\n".join(doc["content"] for doc in outputs.get("documents", []))
    content = f"FACTS: {facts}\nSTUDENT ANSWER: {outputs['answer']}"
    return _grade(GroundedGrade, GROUNDED_INSTRUCTIONS, content)["grounded"]


def _target(inputs: dict) -> dict:
    return ask_question(inputs["question"])


def run_evaluation():
    client = Client()
    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(DATASET_NAME)
        client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)

    return client.evaluate(
        _target,
        data=DATASET_NAME,
        evaluators=[correctness, groundedness],
        experiment_prefix="ytrag-agent",
    )


if __name__ == "__main__":
    run_evaluation()
