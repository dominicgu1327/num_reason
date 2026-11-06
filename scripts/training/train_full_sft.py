import os

os.environ["FLA_TILELANG"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import re
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Qwen3_5ForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


STORAGE_DIR = Path(os.environ.get("NUMREASON_STORAGE", str(Path.home() / "numreason_storage")))
DATA_DIR = STORAGE_DIR / "data_split"
BASE_MODEL_DIR = STORAGE_DIR / "models" / "Qwen3.5-0.8B"
MODEL_DIR = STORAGE_DIR / "models" / "full_sft"
RESULTS_DIR = STORAGE_DIR / "results" / "training" / "full_sft"

SEED = 189
NUM_EPOCHS = 5
BATCH_SIZE = int(os.environ.get("NUMREASON_TRAIN_BATCH", "2"))
EVAL_BATCH_SIZE = int(os.environ.get("NUMREASON_EVAL_BATCH", "4"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("NUMREASON_GRAD_ACCUM", "8"))
LEARNING_RATE = float(os.environ.get("NUMREASON_FULL_LR", "1e-5"))
MAX_LENGTH = 1536

PROMPT = (
    "Solve the following math problem step by step.\n"
    "End your response with the final numerical answer after ####.\n\n"
    "Question: {question}\nAnswer:"
)


def prepare_dataset(dataset, tokenizer):
    def tokenize_example(example):
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

        input_ids = prompt_ids + answer_ids

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + answer_ids,
        }

    processed = dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
    )

    original_size = len(processed)

    processed = processed.filter(
        lambda example: len(example["input_ids"]) <= MAX_LENGTH
    )

    print(f"Examples: {len(processed):,}")
    print(f"Excluded over length limit: {original_size - len(processed):,}")

    return processed


class ValidationHistoryCallback(TrainerCallback):
    def __init__(self):
        self.best_loss = float("inf")
        self.best_epoch = None

        self.previous_loss = None
        self.stopped_early = False
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            raise RuntimeError("Validation loss is unavailable.")

        loss = float(metrics["eval_loss"])
        epoch = int(round(state.epoch))

        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError(f"Non-finite validation loss: {loss}")

        self.history.append({
            "epoch": epoch,
            "validation_loss": round(loss, 6),
        })

        print(f"\n========== EPOCH {epoch} VALIDATION ==========")
        print(f"Validation loss: {loss:.4f}")

        if loss < self.best_loss:
            model = kwargs["model"]

            self.best_loss = loss
            self.best_epoch = epoch

            print(f"New best model at epoch {epoch}")

        if (
            self.previous_loss is not None
            and loss > self.previous_loss
        ):
            self.stopped_early = True
            control.should_training_stop = True

            print(
                f"Early stopping: validation loss increased "
                f"from {self.previous_loss:.4f} to {loss:.4f}"
            )

        self.previous_loss = loss

        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"Best epoch: {self.best_epoch}")
        print("========================================\n")

        return control


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    if not (DATA_DIR / "dataset_dict.json").is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    if not (BASE_MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(f"Model not found: {BASE_MODEL_DIR}")

    set_seed(SEED)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_DIR),
        local_files_only=True,
    )

    tokenizer.padding_side = "right"

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

    # Full fine-tuning: update the original model weights, without PEFT.
    model.requires_grad_(True)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    training_args = TrainingArguments(
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
        warmup_steps=190,
        weight_decay=0.0,
        optim="adamw_torch_fused",
        bf16=True,
        bf16_full_eval=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=20,
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=SEED,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    best_callback = ValidationHistoryCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
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

    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    epochs_completed = len(best_callback.history)

    summary = {
        "method": "full_sft",
        "model_path": str(MODEL_DIR),
        "training_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "max_epochs": NUM_EPOCHS,
        "epochs_completed": epochs_completed,
        "stopped_early": best_callback.stopped_early,
        "best_epoch": best_callback.best_epoch,
        "best_validation_loss": best_callback.best_loss,
        "validation_history": best_callback.history,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percent": round(100 * trainable_params / total_params, 4),
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "train_loss": train_result.training_loss,
        "training_seconds": round(elapsed, 2),
        "training_gpu_hours": round(elapsed / 3600, 4),
        "examples_per_second": round(
            len(train_dataset) * epochs_completed / elapsed, 2
        ),
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
    print(f"Best validation loss: {best_callback.best_loss:.4f}")
    print(f"Training loss: {summary['train_loss']:.4f}")
    print(
        f"Trainable parameters: {trainable_params:,} "
        f"({summary['trainable_percent']}%)"
    )
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training time: {elapsed / 3600:.2f} GPU-hours")
    print(f"Examples/s: {summary['examples_per_second']:.2f}")
    print(f"Peak allocated GPU memory: {peak_memory:.2f} GiB")
    print(f"Peak reserved GPU memory: {peak_reserved:.2f} GiB")
    print(f"Model saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()