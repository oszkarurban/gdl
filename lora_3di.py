"""
lora_3di.py  —  LoRA fine-tuning for inverse folding using Foldseek 3Di tokens.


Usage:
    python lora_3di.py --split Truncated --max_samples 500 --num_epochs 3 \
        --output_dir lora_3di_sanity
"""

import json
import os
import argparse
import numpy as np
import torch
from dataclasses import dataclass

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from inference_3di import load_3di_data, get_split_data, format_3di, build_system_message

IGNORE_INDEX = -100
MAX_LENGTH   = 2048

ALPHABET_3DI = "A C D E F G H I K L M N P Q R S T V W Y"
ALPHABET_AA  = "A C D E F G H I K L M N P Q R S T V W Y"

SPLIT_MAP = {
    'Truncated':    ('chain_set_splits.json', 100),
    'All':          ('chain_set_splits.json', None),
    'Short':        ('test_split_L100.json',  None),
    'Single Chain': ('test_split_sc.json',    None),
}


def build_prompt_and_completion(item: dict, tokenizer, token_format: str) -> tuple:
    """
    Returns (prompt_str, completion_str).
    Prompt is masked in labels (IGNORE_INDEX).
    """
    system_msg = build_system_message(token_format)
    struct_str = format_3di(item['seq_3di'], token_format)

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"3Di structure:\n{struct_str}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    completion = f"{item['seq']}{tokenizer.eos_token}"
    return prompt, completion


class ProteinDesign3DiDataset(Dataset):
    def __init__(self, data_list: list, tokenizer, token_format: str = 'flat'):
        self.data         = data_list
        self.tokenizer    = tokenizer
        self.token_format = token_format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        prompt, completion = build_prompt_and_completion(
            item, self.tokenizer, self.token_format,
        )

        prompt_ids     = self.tokenizer(prompt,     add_special_tokens=False).input_ids
        completion_ids = self.tokenizer(completion, add_special_tokens=False).input_ids

        input_ids = torch.tensor(prompt_ids + completion_ids, dtype=torch.long)
        labels    = torch.tensor(
            [IGNORE_INDEX] * len(prompt_ids) + completion_ids,
            dtype=torch.long,
        )

        input_ids = input_ids[:MAX_LENGTH]
        labels    = labels[:MAX_LENGTH]

        return dict(input_ids=input_ids, labels=labels)



@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [inst[key].clone().detach() for inst in instances]
            for key in ('input_ids', 'labels')
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX,
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def setup_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=MAX_LENGTH,
        padding_side='right',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    return model, tokenizer


def main(args):
    model, tokenizer = setup_model(args)

    split_file, truncate_to = SPLIT_MAP[args.split]

    full_data = load_3di_data(
        os.path.join(args.data_root, 'cath', 'chain_set_3di.jsonl'),
        truncate_to=truncate_to,
    )
    val_list, train_list = get_split_data(
        full_data,
        os.path.join(args.data_root, 'cath', split_file),
        seed=args.seed,
    ) #!!!!Important: val_list first, then train_list

    if args.max_samples is not None:
        train_list = train_list[:args.max_samples]
        val_list   = val_list[:max(1, args.max_samples // 4)]

    print(f"\nSplit       : {args.split} ({split_file})")
    print(f"truncate_to : {truncate_to}")
    print(f"Train       : {len(train_list)} | Val: {len(val_list)}")
    print(f"Token format: {args.token_format}")
    print(f"LR          : {args.lr}\n")

    train_dataset = ProteinDesign3DiDataset(train_list, tokenizer, args.token_format)
    val_dataset   = ProteinDesign3DiDataset(val_list,   tokenizer, args.token_format)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    ex             = train_dataset[0]
    n_masked       = (ex['labels'] == IGNORE_INDEX).sum().item()
    n_supervised   = (ex['labels'] != IGNORE_INDEX).sum().item()
    supervised_ids = ex['labels'][ex['labels'] != IGNORE_INDEX]
    decoded        = tokenizer.decode(supervised_ids)
    print("--- Masking debug (train example 0) ---")
    print(f"  Total tokens       : {len(ex['input_ids'])}")
    print(f"  Masked (prompt)    : {n_masked}")
    print(f"  Supervised (target): {n_supervised}")
    print(f"  Supervised text    : {repr(decoded)}")
    assert decoded.endswith(tokenizer.eos_token), \
        "ERROR: supervised text does not end with eos_token"
    aa_only = decoded.replace(tokenizer.eos_token, '')
    assert all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in aa_only), \
        f"ERROR: supervised text contains non-AA characters: {set(aa_only) - set('ACDEFGHIKLMNPQRSTVWY')}"
    print("  Masking looks correct")
    print("--- End debug ---\n")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        optim='paged_adamw_8bit',
        max_grad_norm=0.3,
        report_to='wandb',
        run_name=(
            f"lora_3di_{args.split}_{args.token_format}"
            f"_r{args.lora_rank}_lr{args.lr}_ep{args.num_epochs}"
        ),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    print(f"\nBest checkpoint : {trainer.state.best_model_checkpoint}")
    print(f"Best eval_loss  : {trainer.state.best_metric:.4f}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved → {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',   default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--data_root',    default='data/')
    parser.add_argument('--output_dir',   default='lora_3di_adapter')
    parser.add_argument('--split',        default='Truncated',
                        choices=['Truncated', 'Short', 'All', 'Single Chain'])
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--max_samples',  type=int,   default=None)
    parser.add_argument('--token_format', default='flat', choices=['flat', 'spaced'])
    parser.add_argument('--num_epochs',   type=int,   default=10)
    parser.add_argument('--batch_size',   type=int,   default=1)
    parser.add_argument('--grad_accum',   type=int,   default=4)
    parser.add_argument('--lr',           type=float, default=2e-5,
                        help='Safe default for instruct model. Try 5e-5 if loss plateaus.')
    parser.add_argument('--lora_rank',    type=int,   default=16)
    parser.add_argument('--lora_alpha',   type=int,   default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    args = parser.parse_args()

    main(args)