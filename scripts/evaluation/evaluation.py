
import csv
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_from_disk
from peft import PeftConfig, PeftModel, PeftType
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


STORAGE_DIR = Path("/Volumes/workspace/default/numreason")
DATA_DIR = STORAGE_DIR / "data_split"
BASE_MODEL_DIR = STORAGE_DIR / "models" / "Qwen3.5-0.8B"
RESULTS_DIR = STORAGE_DIR / "results" / "evaluation"

BATCH_SIZE = 32
MAX_NEW_TOKENS = 512
SEED = 189

PROMPT = (
    "Solve the following math problem step by step.\n"
    "End your response with the final numerical answer after ####.\n\n"
    "Question: {question}\nAnswer:"
)

NUMBER_PATTERN = re.compile(
    r"\s*\$?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\.?\s*"
)


def parse_answer(text):
    if "####" not in text:
        return None

    final_text = text.rsplit("####", 1)[-1]
    match = NUMBER_PATTERN.fullmatch(final_text)

    if match is None:
        return None

    try:
        number = Decimal(match.group(1).replace(",", ""))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def load_model(model_path):
    if not BASE_MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_DIR),
        local_files_only=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dir = BASE_MODEL_DIR
    adapter_type = None

    if model_path is not None:
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if (model_path / "adapter_config.json").is_file():
            config = PeftConfig.from_pretrained(str(model_path))
            adapter_type = config.peft_type

            if config.task_type != "CAUSAL_LM":
                raise ValueError("Expected a CAUSAL_LM adapter.")
        else:
            model_dir = model_path

    options = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else "cpu",
        "local_files_only": True,
    }

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        **options,
    )

    if adapter_type is not None:
        model = PeftModel.from_pretrained(
            model,
            str(model_path),
            is_trainable=False,
        )

    model.eval()

    is_soft_prompt = adapter_type == PeftType.PROMPT_TUNING

    return model, tokenizer, is_soft_prompt


def make_prompt(question, tokenizer):
    messages = [{
        "role": "user",
        "content": PROMPT.format(question=question),
    }]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def evaluate(args):
    if args.limit is not None and args.limit <= 0:
        raise ValueError("Limit must be positive.")

    if not (DATA_DIR / "dataset_dict.json").is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    dataset = load_from_disk(str(DATA_DIR))["test"]

    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    if len(dataset) == 0:
        raise ValueError("No examples selected.")

    torch.manual_seed(SEED)

    model, tokenizer, is_soft_prompt = load_model(
        args.model_path,
    )

    device = next(model.parameters()).device

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    rows = []
    start = time.perf_counter()

    for offset in tqdm(
        range(0, len(dataset), BATCH_SIZE),
        desc="Evaluating",
        unit="batch",
    ):
        batch = dataset.select(
            range(offset, min(offset + BATCH_SIZE, len(dataset)))
        )

        prompts = [
            make_prompt(question, tokenizer)
            for question in batch["question"]
        ]

        inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=not is_soft_prompt,
            )

        prompt_width = inputs["input_ids"].shape[1]
        generated = []

        for index, output in enumerate(outputs):
            original = inputs["input_ids"][index]

            if torch.equal(output[:prompt_width], original):
                tokens = output[prompt_width:]
            elif is_soft_prompt:
                tokens = output
            else:
                raise RuntimeError("Unexpected generated token format.")

            generated.append(tokens)

        responses = [
            tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in generated
        ]

        for question, solution, response in zip(
            batch["question"],
            batch["answer"],
            responses,
        ):
            reference = parse_answer(solution)
            prediction = parse_answer(response)

            if reference is None:
                raise ValueError(f"Invalid reference answer: {solution}")

            rows.append({
                "question": question,
                "reference": str(reference),
                "prediction": str(prediction) if prediction is not None else "",
                "correct": int(prediction == reference),
                "response": response,
            })

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    correct = sum(row["correct"] for row in rows)
    parsed = sum(bool(row["prediction"]) for row in rows)

    summary = {
        "model": args.name,
        "model_path": str(args.model_path) if args.model_path else str(BASE_MODEL_DIR),
        "examples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "answer_parse_rate": parsed / len(rows),
        "evaluation_seconds": round(elapsed, 2),
        "peak_inference_memory_gib": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            if torch.cuda.is_available() else None
        ),
    }

    output_dir = RESULTS_DIR / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAccuracy: {summary['accuracy']:.2%}")
    print(f"Correct answers: {correct}/{len(rows)}")
    print(f"Answer parse rate: {summary['answer_parse_rate']:.2%}")
    print(f"Evaluation time: {elapsed:.2f} seconds")

    if summary["peak_inference_memory_gib"] is not None:
        print(
            f"Peak inference GPU memory: "
            f"{summary['peak_inference_memory_gib']:.2f} GiB"
        )

    print(f"Results saved to: {output_dir}")


def main():
    params = dbutils.widgets.getAll()

    args = SimpleNamespace(
        name=params.get("name", "baseline"),
        model_path=(
            Path(params["model_path"])
            if params.get("model_path")
            else None
        ),
        limit=(
            int(params["limit"])
            if params.get("limit")
            else None
        ),
    )

    evaluate(args)


if __name__ == "__main__":
    main()