# NumReason: Post-Training Qwen for Reliable Numerical Reasoning

NumReason is a research project that investigates how different post-training methods improve the mathematical reasoning capabilities of small language models.

We use **Qwen3.5-0.8B** as the base model and **GSM8K** as the primary benchmark. The project compares full-parameter fine-tuning, LoRA, QLoRA, and other parameter-efficient approaches under a consistent evaluation framework.

## Project Overview

Small language models are computationally efficient but often struggle with multi-step numerical reasoning. NumReason investigates whether supervised fine-tuning (SFT) and reinforcement learning with Group Relative Policy Optimization (GRPO) can improve their performance.

Our research focuses on three questions:

1. How do different fine-tuning methods compare in numerical reasoning accuracy?
    
2. How does parameter efficiency affect performance and computational cost?

## Model and Dataset

**Base model:** Qwen3.5-0.8B

**Dataset:** [GSM8K](https://huggingface.co/datasets/openai/gsm8k)

GSM8K contains grade-school mathematics word problems that require multi-step reasoning.

- Training set: 7,473 problems
    
- Test set: 1,319 problems

The model is trained to generate step-by-step solutions and produce a final numerical answer.

## Methods

We explore the following post-training strategies:

**Training Objectives**

- Supervised Fine-Tuning (SFT)
    
- Group Relative Policy Optimization (GRPO)
    
- SFT → GRPO

**Fine-Tuning Methods**

- Full Fine-Tuning
    
- LoRA
    
- QLoRA
    
- Prompt Tuning
## Setup

The project is developed using Python, with experiments run on Databricks GPU compute.

Install dependencies:

`pip install -r requirements.txt`

Configure the dataset paths, model paths, and output directories in the scripts for your environment.

## Data Preparation

Download the GSM8K dataset:

`python scripts/data/download_data.py`

Process the dataset and prepare the training, validation, and test splits:

`python scripts/data/process_data.py`

## Training

The repository provides separate scripts for different SFT strategies.

**Full fine-tuning**

`python scripts/training/train_full_sft.py`

**LoRA fine-tuning**

`python scripts/training/train_lora_sft.py`

**QLoRA fine-tuning**

`python scripts/training/train_qlora_sft.py`

**Prompt tuning**

`python scripts/training/train_prompt_tuning_sft.py`

These commands assume that the scripts' configured model, dataset, and output paths are accessible.

## Evaluation

Evaluate the model using:

`python scripts/evaluation/evaluation.py`

The primary evaluation metric is **final-answer accuracy** on GSM8K.

Predicted numerical answers are extracted from generated responses and compared against the ground-truth answers.

## Acknowledgments

This project is developed for **CS 289A: Introduction to Machine Learning** at the University of California, Berkeley.

Experiments use Databricks GPU computing resources.
