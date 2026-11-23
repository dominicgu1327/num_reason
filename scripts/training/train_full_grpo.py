import os

os.environ["FLA_TILELANG"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Qwen3_5ForCausalLM,
    TrainerCallback,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer


STORAGE_DIR = Path("/Volumes/workspace/default/numreason")
DATA_DIR = STORAGE_DIR / "data_split"
BASE_MODEL_DIR = STORAGE_DIR / "models" / "Qwen3.5-0.8B"
MODEL_DIR = STORAGE_DIR / "models" / "full_grpo"
RESULTS_DIR = STORAGE_DIR / "results" / "training" / "full_grpo"

SEED = 189
NUM_EPOCHS = 1
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5
MAX_LENGTH = 1536

NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 512
TEMPERATURE = 1.0
BETA = 0.0

PROMPT = (
    "Solve the following math problem step by step.\n"
    "End your response with the final numerical answer after ####.\n\n"
    "Question: {question}\nAnswer:"
)


def prepare_dataset(dataset, tokenizer):
    def process_example(example):
        messages = [{
            "role": "user",
            "content": PROMPT.format(question=example["question"]),
        }]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        solution = re.sub(
            r"<<.*?>>",
            "",
            example["answer"],
            flags=re.DOTALL,
        ).strip()

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        answer_ids = tokenizer(
            solution + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]

        return {
            "prompt": prompt,
            "ground_truth": example["answer"],
            "total_length": len(prompt_ids) + len(answer_ids),
        }

    processed = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
    )

    original_size = len(processed)

    processed = processed.filter(
        lambda example: example["total_length"] <= MAX_LENGTH
    )

    processed = processed.remove_columns(["total_length"])

    print(f"Examples: {len(processed):,}")
    print(f"Excluded over length limit: {original_size - len(processed):,}")

    return processed


def extract_answer(text):
    match = re.search(
        r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*$",
        text.strip(),
    )

    if match is None:
        return None

    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None


def accuracy_reward(completions, ground_truth, **kwargs):
    rewards = []

    for completion, expected in zip(completions, ground_truth):
        predicted_answer = extract_answer(completion)
        correct_answer = extract_answer(expected)

        reward = float(
            predicted_answer is not None
            and correct_answer is not None
            and predicted_answer == correct_answer
        )

        rewards.append(reward)

    return rewards


class BestModelCallback(TrainerCallback):
    def __init__(self):
        self.best_reward = float("-inf")
        self.best_epoch = None

        self.previous_reward = None
        self.stopped_early = False
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            raise RuntimeError("Validation metrics are unavailable.")

        reward = metrics.get(
            "eval_reward",
            metrics.get("eval_rewards/accuracy_reward/mean"),
        )

        if reward is None:
            raise RuntimeError(
                f"Validation reward is unavailable. Metrics: {list(metrics)}"
            )

        reward = float(reward)
        epoch = int(round(state.epoch))

        if not torch.isfinite(torch.tensor(reward)):
            raise RuntimeError(f"Non-finite validation reward: {reward}")

        self.history.append({
            "epoch": epoch,
            "validation_reward": round(reward, 6),
        })

        print(f"\n========== EPOCH {epoch} VALIDATION ==========")
        print(f"Validation reward: {reward:.4f}")

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_epoch = epoch

            print(f"New best model at epoch {epoch}")

        if (
            self.previous_reward is not None
            and reward < self.previous_reward
        ):
            self.stopped_early = True
            control.should_training_stop = True

            print(
                f"Early stopping: validation reward decreased "
                f"from {self.previous_reward:.4f} to {reward:.4f}"
            )

        self.previous_reward = reward

        print(f"Best validation reward: {self.best_reward:.4f}")
        print(f"Best epoch: {self.best_epoch}")
        print("========================================\n")

        return control


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    if not (DATA_DIR / "dataset_dict.json").is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    if not (BASE_MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL_DIR}")

    set_seed(SEED)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_DIR),
        local_files_only=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(str(DATA_DIR))

    train_dataset = prepare_dataset(dataset["train"], tokenizer)
    validation_dataset = prepare_dataset(dataset["validation"], tokenizer)

    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training or validation dataset is empty.")

    model = Qwen3_5ForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )

    model.config.use_cache = False

    for param in model.parameters():
        param.requires_grad_(True)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found.")

    print(
        f"Trainable parameters: {trainable_params:,} "
        f"({100 * trainable_params / total_params:.4f}%)"
    )

    training_args = GRPOConfig(
        output_dir=str(RESULTS_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=40,
        weight_decay=0.0,
        optim="adamw_torch_fused",
        bf16=True,
        bf16_full_eval=True,
        tf32=True,
        gradient_checkpointing=False,
        num_generations=NUM_GENERATIONS,
        num_generations_eval=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=TEMPERATURE,
        beta=BETA,
        loss_type="dapo",
        scale_rewards="group",
        use_vllm=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_reward",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=False,
        seed=SEED,
    )

    best_callback = BestModelCallback()

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=accuracy_reward,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        callbacks=[best_callback],
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    train_result = trainer.train()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    if best_callback.best_epoch is None:
        raise RuntimeError("No best model was selected.")

    if trainer.state.best_model_checkpoint is None:
        raise RuntimeError("Best model checkpoint is unavailable.")

    # The Trainer restores the best full-model weights automatically.
    model.config.use_cache = True

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    epochs_completed = len(best_callback.history)

    summary = {
        "method": "full_grpo",
        "initial_model_path": str(BASE_MODEL_DIR),
        "model_path": str(MODEL_DIR),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "training_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "max_epochs": NUM_EPOCHS,
        "epochs_completed": epochs_completed,
        "stopped_early": best_callback.stopped_early,
        "best_epoch": best_callback.best_epoch,
        "best_validation_reward": best_callback.best_reward,
        "validation_history": best_callback.history,
        "num_generations": NUM_GENERATIONS,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "temperature": TEMPERATURE,
        "beta": BETA,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percent": round(
            100 * trainable_params / total_params, 4
        ),
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "train_loss": train_result.training_loss,
        "global_steps": trainer.state.global_step,
        "training_seconds": round(elapsed, 2),
        "training_gpu_hours": round(elapsed / 3600, 4),
        "peak_training_memory_gib": round(peak_memory, 2),
        "peak_reserved_memory_gib": round(peak_reserved, 2),
    }

    with (RESULTS_DIR / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"\nEpochs completed: {epochs_completed}/{NUM_EPOCHS}")
    print(f"Stopped early: {best_callback.stopped_early}")
    print(f"Best epoch: {best_callback.best_epoch}")
    print(f"Best validation reward: {best_callback.best_reward:.4f}")
    print(f"Training loss: {summary['train_loss']:.4f}")
    print(
        f"Trainable parameters: {trainable_params:,} "
        f"({summary['trainable_percent']}%)"
    )
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training time: {elapsed / 3600:.2f} GPU-hours")
    print(f"Peak allocated GPU memory: {peak_memory:.2f} GiB")
    print(f"Peak reserved GPU memory: {peak_reserved:.2f} GiB")
    print(f"Model saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
