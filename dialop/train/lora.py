import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import wandb
from datasets import Dataset, load_from_disk
from fire import Fire
from unsloth import FastLanguageModel
# from peft import LoraConfig, get_peft_model
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    # AutoModelForCausalLM,
    # AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataset(
    dataset: Dataset,
    # tokenizer: AutoTokenizer,
    tokenizer,
    max_seq_length: int,
) -> Dataset:
    def extract_chat_components(messages: List[Dict[str, str]]) -> Tuple[str, str, str]:
        system_msg = ""
        user_msg = ""
        assistant_msg = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                system_msg = content
            elif role == "user":
                user_msg = content
            elif role == "assistant":
                assistant_msg = content
        return system_msg, user_msg, assistant_msg

    def _map_fn(batch: Dict[str, List]) -> Dict[str, List]:
        input_ids_list: List[List[int]] = []
        attention_mask_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        system_list: List[str] = []
        user_list: List[str] = []
        assistant_list: List[str] = []
        prompt_messages_list: List[List[Dict[str, str]]] = []
        full_sequence_lengths: List[int] = []

        for messages in batch["messages"]:
            system_msg, user_msg, assistant_msg = extract_chat_components(messages)

            prompt_messages: List[Dict[str, str]] = []
            if system_msg:
                prompt_messages.append({"role": "system", "content": system_msg})
            prompt_messages.append({"role": "user", "content": user_msg})

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_length,
            )["input_ids"]

            assistant_with_eos = assistant_msg + tokenizer.eos_token
            full_text = prompt_text + assistant_with_eos
            tokenized = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_length,
            )

            labels = list(tokenized["input_ids"])
            prompt_len = min(len(prompt_ids), len(labels))
            for idx in range(prompt_len):
                labels[idx] = -100

            input_ids_list.append(tokenized["input_ids"])
            attention_mask_list.append(tokenized["attention_mask"])
            labels_list.append(labels)
            system_list.append(system_msg)
            user_list.append(user_msg)
            assistant_list.append(assistant_msg)
            prompt_messages_list.append(prompt_messages)
            full_sequence_lengths.append(len(tokenizer.tokenize(full_text)))

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
            "system": system_list,
            "user": user_list,
            "assistant": assistant_list,
            "prompt_messages": prompt_messages_list,
            "full_sequence_length": full_sequence_lengths,
        }

    return dataset.map(
        _map_fn,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in {"messages"}],
        load_from_cache_file=False,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Batch:
    def _to_long(t):
        return t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long)

    input_ids = pad_sequence(
        [_to_long(item["input_ids"]) for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [_to_long(item["attention_mask"]) for item in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [_to_long(item["labels"]) for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def compute_param_norm(parameters: List[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.requires_grad:
            total += torch.sum(p.detach() ** 2).item()
    return math.sqrt(total)


def evaluate(
    # model: AutoModelForCausalLM,
    model: FastLanguageModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss.detach()
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        return float("nan"), float("nan")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))
    return avg_loss, perplexity


def qualitative_sampling(
    model: FastLanguageModel,
    # tokenizer: AutoTokenizer,
    tokenizer,
    val_samples: List[Dict],
    device: torch.device,
    args,
    epoch: int,
) -> None:
    rng = random.Random(args.seed + epoch)
    sampled = rng.sample(val_samples, k=min(5, len(val_samples)))
    table = wandb.Table(columns=["system", "user", "reference", "generated"])
    total_words = 0

    model.eval()
    for example in sampled:
        prompt_messages = example["prompt_messages"]
        reference = example["assistant"]

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            generated_ids[0][prompt_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        total_words += len(generated.split())
        system_text = next((m["content"] for m in prompt_messages if m["role"] == "system"), "")
        user_text = next((m["content"] for m in prompt_messages if m["role"] == "user"), "")
        table.add_data(system_text, user_text, reference, generated)

    wandb.log({
        "eval/qualitative_samples": table,
        "eval/generated_word_count": total_words,
    })


class TrainConfig:
    def __init__(
        self,
        dataset_path: str = str(
            Path(__file__).parent.parent / "data" / "sft_datasets" / "optimization_human-human"
        ),
        output_dir: str = str(Path(__file__).parent.parent / "checkpoints" / "lora"),
        val_ratio: float = 0.3,
        seed: int = 42,
        num_epochs: int = 10,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.03,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        max_seq_length: int = 100000,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        project: str = "assistance_sft_matching",
        entity: str = None,
        log_every: int = 10,
    ) -> None:
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.val_ratio = val_ratio
        self.seed = seed
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.max_seq_length = max_seq_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.project = project
        self.entity = entity
        self.log_every = log_every


def main(**kwargs) -> None:
    args = TrainConfig(**kwargs)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
        "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
        "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    if device.type == "cuda":
        model = model.to(device)

    raw_dataset = load_from_disk(args.dataset_path)
    split_dataset = raw_dataset.train_test_split(
        test_size=args.val_ratio,
        seed=args.seed,
        shuffle=True,
    )
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    train_dataset = prepare_dataset(train_dataset, tokenizer, args.max_seq_length)
    val_dataset = prepare_dataset(val_dataset, tokenizer, args.max_seq_length)

    val_samples_for_generation = [val_dataset[i] for i in range(len(val_dataset))]

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    total_trainable_params = [p for p in model.parameters() if p.requires_grad]

    total_optimization_steps = (
        math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        * args.num_epochs
    )
    optimizer = torch.optim.AdamW(
        total_trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_optimization_steps),
        num_training_steps=total_optimization_steps,
    )

    if device.type == "cuda":
        scaler = torch.amp.GradScaler(device="cuda")
    else:
        scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    wandb.init(
        project=args.project,
        entity=args.entity,
        config={
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
    )

    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader, start=1):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda")
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if step % args.gradient_accumulation_steps == 0:
                if device.type == "cuda":
                    scaler.unscale_(optimizer)

                grad_norm = clip_grad_norm_(
                    total_trainable_params,
                    args.max_grad_norm,
                )
                param_norm = compute_param_norm(total_trainable_params)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                loss_value = outputs.loss.detach().item()
                train_perplexity = math.exp(min(loss_value, 100))

                if global_step % args.log_every == 0:
                    wandb.log(
                        {
                            "train/loss": loss_value,
                            "train/perplexity": train_perplexity,
                            "train/grad_norm": grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else float(grad_norm),
                            "train/param_norm": param_norm,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/step": global_step,
                            "epoch": epoch,
                        }
                    )

        eval_loss, eval_ppl = evaluate(model, val_dataloader, device)
        wandb.log(
            {
                "eval/loss": eval_loss,
                "eval/perplexity": eval_ppl,
                "epoch": epoch,
            }
        )

        qualitative_sampling(
            model,
            tokenizer,
            val_samples_for_generation,
            device,
            args,
            epoch,
        )

        output_path = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    wandb.finish()


if __name__ == "__main__":
    Fire(main)