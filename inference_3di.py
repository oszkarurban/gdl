"""
inference_3di.py   run inverse folding using Foldseek 3Di tokens.

Usage:
# Zero-shot
python inference_3di.py --split Truncated --max_samples 10 \
    --output_file results_3di_zeroshot.json

# Few-shot
python inference_3di.py --split Truncated --max_samples 10 --few_shot_k 3 \
    --output_file results_3di_3shot.json

# With loading finetuned (LoRA) adapter
python inference_3di.py --split Truncated --max_samples 10\
    --lora_adapter lora_3di_adapter \
    --output_file results_3di_lora.json

"""

import json
import os
import numpy as np
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',   default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--data_root',    default='data/')
    parser.add_argument('--split',        default='Truncated',
                        choices=['Truncated', 'Short', 'All', 'Single Chain'])
    parser.add_argument('--few_shot_k',   default=0, type=int)
    parser.add_argument('--output_file',  default='results_3di.json')
    parser.add_argument('--lora_adapter', default=None)
    parser.add_argument('--seed',         default=42, type=int)
    parser.add_argument('--max_samples',  default=None, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--token_format', default='flat', choices=['flat', 'spaced']) #whether to include spaces between 3di tokens
    parser.add_argument('--temperature',  default=0.7, type=float)
    parser.add_argument('--top_p',        default=0.9, type=float)
    return parser.parse_args()



# Foldseek 3Di alphabet — 20 letters, same characters (afaik) as AA
ALPHABET_3DI = "A C D E F G H I K L M N P Q R S T V W Y"
# Output amino acid alphabet
ALPHABET_AA  = "A C D E F G H I K L M N P Q R S T V W Y"

# Split map: split_name → (split_file, truncate_to)
# 'Truncated' = same proteins as 'All' but sequences sliced to 100 residues
# 'Short'     = test_split_L100.json proteins are natively short (≤100), no truncation needed
SPLIT_MAP = {
    'Truncated':    ('chain_set_splits.json', 100),
    'All':          ('chain_set_splits.json', None),
    'Short':        ('test_split_L100.json',  None),
    'Single Chain': ('test_split_sc.json',    None),
}

def load_3di_data(jsonl_path: str, truncate_to: int = None) -> list:
    """
    Load chain_set_3di.jsonl produced by jsonl_to_3di.py.
    """
    alphabet_set = set('ACDEFGHIKLMNPQRSTVWY')
    data_list = []

    with open(jsonl_path) as f:
        for line in f:
            entry   = json.loads(line)
            seq     = entry['seq']
            seq_3di = entry['seq_3di']

            # checks (should always pass)
            if len(seq) != len(seq_3di):
                continue
            if len(set(seq) - alphabet_set) > 0:
                continue

            if truncate_to and len(seq) > truncate_to:
                seq     = seq[:truncate_to]
                seq_3di = seq_3di[:truncate_to]

            data_list.append({
                'title':   entry['name'],
                'seq':     seq,
                'seq_3di': seq_3di,
                'CATH':    entry.get('CATH', []),
            })

    return data_list


def get_split_data(full_data: list, split_file: str, seed: int = 42) -> tuple:
    """
    Returns (test_list, train_list).

    chain_set_splits.json  → has 'train' + 'test' keys
    test_split_L100.json   → has only 'test' key  (Short split)
    test_split_sc.json     → has only 'test' key  (Single Chain split)

    For splits without a train key: 20% train / 80% test from the subset. (the reason for this "unconventional" split is that we want to eval more than train, but if it is a problem i can flip it)
    """
    with open(split_file) as f:
        split_data = json.load(f)

    if 'train' in split_data:
        test_names  = set(split_data['test'])
        train_names = set(split_data['train'])
        test_list   = [d for d in full_data if d['title'] in test_names]
        train_list  = [d for d in full_data if d['title'] in train_names]
    else:
        subset = [d for d in full_data if d['title'] in set(split_data['test'])]
        rng = np.random.RandomState(seed)
        rng.shuffle(subset)
        split_idx  = max(1, int(len(subset) * 0.2))
        train_list = subset[:split_idx]
        test_list  = subset[split_idx:]
        print(f"  No train split — 20/80: {len(train_list)} train, {len(test_list)} test")

    return test_list, train_list   # (test, train)



def format_3di(seq_3di: str, token_format: str) -> str:
    if token_format == 'flat':
        return seq_3di                  # "DVQTAGNF..."
    elif token_format == 'spaced':
        return ' '.join(seq_3di)        # "D V Q T A G N F ..."


def build_system_message(token_format: str) -> str:
    if token_format == 'flat':
        fmt_desc = 'a compact string with no spaces (e.g. "DVQTAGNF...")'
    else:
        fmt_desc = 'space-separated tokens (e.g. "D V Q T A G N F ...")'

    return (
        "You are an expert in computational structural biology.\n"
        "Your task is to recover the amino acid sequence of a protein from its 3Di structure tokens.\n"
        "3Di is a 20-letter discrete structural alphabet computed by Foldseek. "
        "Each token encodes the local geometric environment of one residue "
        "(backbone angles, virtual CB orientation, nearest-neighbour geometry). "
        "It is rotation and translation invariant — one token per residue.\n"
        f"The structure is provided as {fmt_desc}, one token per residue.\n"
        f"3Di input alphabet  (20 tokens) : {ALPHABET_3DI}\n"
        f"Output amino acid alphabet (20) : {ALPHABET_AA}\n"
        "IMPORTANT: Both alphabets use the same 20 letters but with completely different meanings. "
        "3Di tokens describe local geometry; amino acids describe chemistry. "
        "Do NOT copy the input - predict the correct amino acid for each position.\n"
        "Output ONLY the amino acid sequence - a single string with no spaces, "
        "punctuation, or explanation."
    )


def get_prompt(item: dict, examples: list, token_format: str) -> str:
    system_msg = build_system_message(token_format)

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
    )

    for ex in examples:
        struct_str = format_3di(ex['seq_3di'], token_format)
        prompt += (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"3Di structure:\n{struct_str}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{ex['seq']}<|eot_id|>"
        )

    target_str = format_3di(item['seq_3di'], token_format)
    prompt += (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"3Di structure:\n{target_str}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        # model generates from here
    )
    return prompt



def calculate_recovery(true_seq: str, pred_seq: str) -> float:
    """Fraction of positions correctly predicted. Penalises short predictions."""
    n = min(len(true_seq), len(pred_seq))
    if n == 0:
        return 0.0
    matches = sum(t == p for t, p in zip(true_seq, pred_seq))
    return matches / len(true_seq)


def is_copying_input(seq_3di: str, prediction: str) -> bool:
    """Detect if model is just echoing the 3Di input"""
    n = min(len(seq_3di), len(prediction))
    if n == 0:
        return False
    matches = sum(a == b for a, b in zip(seq_3di, prediction))
    return (matches / n) > 0.8


# def clean_prediction(raw: str) -> str:
#     # previously i was filtering
#     return raw

def clean_prediction(raw: str) -> str:
    alphabet = set('ACDEFGHIKLMNPQRSTVWY')
    tokens = raw.strip().split()
    candidate = tokens[0] if tokens else ''
    if all(c in alphabet for c in candidate.upper()):
        return candidate.upper()
    return ''.join(c for c in raw.upper() if c in alphabet)


def main():
    args = get_parser()
    rng  = np.random.RandomState(args.seed)

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    if args.lora_adapter:
        print(f"Loading LoRA adapter: {args.lora_adapter}")
        model = PeftModel.from_pretrained(model, args.lora_adapter)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    split_file, truncate_to = SPLIT_MAP[args.split]

    full_data = load_3di_data(
        os.path.join(args.data_root, 'cath', 'chain_set_3di.jsonl'),
        truncate_to=truncate_to,
    )
    test_data, train_data = get_split_data(
        full_data,
        os.path.join(args.data_root, 'cath', split_file),
        seed=args.seed,
    )

    print(f"Split: {args.split} | truncate_to: {truncate_to} | "
          f"Train: {len(train_data)} | Test: {len(test_data)}")

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    results = []
    print(f"\nToken format: {args.token_format} | Temperature: {args.temperature}\n")

    for item in tqdm(test_data):
        examples = []
        if args.few_shot_k > 0 and train_data:
            idx      = rng.choice(len(train_data), min(args.few_shot_k, len(train_data)), replace=False)
            examples = [train_data[i] for i in idx]

        prompt = get_prompt(item, examples, token_format=args.token_format)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

        max_new_tokens = len(item['seq']) + 0 #previously tried (20)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["<|eot_id|>"],
                tokenizer=tokenizer,
            )

        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        raw_pred      = tokenizer.decode(generated_ids, skip_special_tokens=True)
        prediction    = clean_prediction(raw_pred)
        recovery      = calculate_recovery(item['seq'], prediction)
        copying       = is_copying_input(item['seq_3di'], prediction)

        print(f"\n{item['title']}")
        print(f"  3Di    : {item['seq_3di'][:50]}")
        print(f"  Target : {item['seq']}")
        print(f"  Pred   : {prediction}")
        print(f"  Recovery: {recovery:.4f}  {'[COPYING INPUT]' if copying else ''}")

        results.append({
            'title':         item['title'],
            'target':        item['seq'],
            'seq_3di':       item['seq_3di'],
            'prediction':    prediction,
            'recovery':      recovery,
            'copying_input': copying,
        })

    avg_recovery  = float(np.mean([r['recovery']      for r in results]))
    copy_rate     = float(np.mean([r['copying_input']  for r in results]))

    print(f"\n{'='}")
    print(f"Average Sequence Recovery : {avg_recovery:.4f}")
    print(f"Input-copying rate        : {copy_rate:.3f}  (should be ~0.0)")
    print(f"{'='}")

    out = {
        'args':         vars(args),
        'avg_recovery': avg_recovery,
        'copy_rate':    copy_rate,
        'results':      results,
    }
    with open(args.output_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {args.output_file}")


if __name__ == '__main__':
    main()