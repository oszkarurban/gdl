# """
# filteredinference.py
# --------------
# Evaluate LLM checkpoints on CATH inverse folding.

# Usage:
#     python filteredinference.py --checkpoint results/llm_3di --test_file data/tokenized3/test.jsonl
#     python filteredinference.py --zero_shot --test_file data/tokenized3/test.jsonl
#     python filteredinference.py \
#         --checkpoint results/llm_3di_n500 results/llm_3di \
#         --labels "n=500" "n=full" \
#         --test_file data/tokenized3/test.jsonl
# """

# import json, os, math, random, argparse
# import torch
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel, PeftConfig

# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# STRUCT_BOS = "<struct>"
# SEQ_BOS    = "<seq>"
# AA_VOCAB   = set("ACDEFGHIKLMNPQRSTVWY")
# AA_CHARS   = list("ACDEFGHIKLMNPQRSTVWY")
# DI3_CHARS  = list("acdefghiklmnpqrstvwy")


# def make_prompt(tokens_3di):
#     return f"{STRUCT_BOS}{tokens_3di}{SEQ_BOS}"


# def setup_tokenizer(path):
#     tok = AutoTokenizer.from_pretrained(path)
#     tok.pad_token = tok.eos_token
#     tok.add_special_tokens({"additional_special_tokens": [STRUCT_BOS, SEQ_BOS]})
#     new = [t for t in AA_CHARS + DI3_CHARS if t not in tok.get_vocab()]
#     if new:
#         tok.add_tokens(new)
#     return tok


# def load_base(model_name, tokenizer):
#     """Load base model on CPU, resize, cast to bf16, move to GPU."""
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True,
#     )
#     model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
#     return model.to(torch.bfloat16).to("cuda").eval()


# def load_finetuned(checkpoint):
#     cfg = PeftConfig.from_pretrained(checkpoint)
#     # Tokenizer: checkpoint dir → parent → base model
#     tok_path = next(
#         (p for p in [checkpoint, os.path.dirname(checkpoint), cfg.base_model_name_or_path]
#          if os.path.exists(os.path.join(p, "tokenizer_config.json"))),
#         cfg.base_model_name_or_path,
#     )
#     tokenizer = setup_tokenizer(tok_path)
#     model     = load_base(cfg.base_model_name_or_path, tokenizer)
#     model     = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
#     print(f"  ✓ LoRA merged from {checkpoint}")
#     return model.eval(), tokenizer


# def load_zeroshot(model_name):
#     tokenizer = setup_tokenizer(model_name)
#     model     = load_base(model_name, tokenizer)
#     print(f"  ✓ Zero-shot {model_name}")
#     return model, tokenizer

# @torch.no_grad()
# def compute_recovery(model, tokenizer, records, max_length):
#     correct, total, examples = 0, 0, []

#     for r in tqdm(records, desc="Recovery"):
#         inputs = tokenizer(make_prompt(r["tokens_3di"]), return_tensors="pt",
#                            truncation=True, max_length=max_length).to(model.device)
#         out = model.generate(
#             **inputs,
#             max_new_tokens=r["length"],
#             do_sample=False,
#             temperature=None,
#             top_p=None,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#         )
#         gen  = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#         pred = "".join(c for c in gen if c in AA_VOCAB)[:r["length"]]
#         print(f"gen: {gen} \n pred: {pred}")
#         ref  = r["seq"]
#         print(f"ref: {ref}")   
#         c    = sum(p == q for p, q in zip(pred, ref))
#         correct += c; total += len(ref)
#         examples.append({
#             "name": r["name"], "length": len(ref), "pred_len": len(pred),
#             "ref": ref, "pred": pred, "recovery": c / len(ref),
#         })
#     return correct / total if total else 0.0, examples


# def print_results(label, recovery, examples):
#     def wavg(sub):
#         return sum(e["recovery"]*e["length"] for e in sub) / sum(e["length"] for e in sub) \
#                if sub else float("nan")
#     short  = [e for e in examples if e["length"] <= 100]
#     medium = [e for e in examples if 100 < e["length"] <= 300]
#     long_  = [e for e in examples if e["length"] > 300]
#     exact  = sum(1 for e in examples if e["pred_len"] == e["length"])

#     print(f"\n{'='*55}\n  {label}  (n={len(examples)})\n{'='*55}")
#     print(f"  Recovery (all)    : {recovery*100:6.2f}%")
#     print(f"  Recovery (≤100aa) : {wavg(short)*100:6.2f}%  (n={len(short)})")
#     print(f"  Recovery (101-300): {wavg(medium)*100:6.2f}%  (n={len(medium)})")
#     print(f"  Recovery (>300aa) : {wavg(long_)*100:6.2f}%  (n={len(long_)})")
#     print(f"  Exact length      : {exact}/{len(examples)}")
#     print(f"  Baselines: ProteinMPNN=52.4%  ESM-IF=51.6%")
#     for e in examples[:3]:
#         flag = "✓" if e["pred_len"] == e["length"] else f"len={e['pred_len']}"
#         print(f"\n  {e['name']} rec={e['recovery']:.3f} {flag}")
#         print(f"    ref : {e['ref'][:60]}")
#         print(f"    pred: {e['pred'][:60]}")


# def main(args):
#     random.seed(42)
#     records = [json.loads(l) for l in open(args.test_file) if l.strip()]
#     records = [r for r in records if r.get("tokens_3di") and r.get("seq")]
#     if args.n_eval > 0:
#         records = random.sample(records, min(args.n_eval, len(records)))
#     print(f"Evaluating {len(records)} chains")

#     all_results = {}

#     def run(model, tokenizer, label):
#         recovery, exs  = compute_recovery(model, tokenizer, records, args.max_length)
#         print_results(label, recovery, exs)
#         all_results[label] = {"recovery": recovery, "n_eval": len(records), "examples": exs[:10]}
#         del model; torch.cuda.empty_cache()

#     if args.zero_shot:
#         m, t = load_zeroshot(args.model_name)
#         run(m, t, f"zero-shot ({args.model_name.split('/')[-1]})")

#     labels = args.labels or args.checkpoint or []
#     for ckpt, label in zip(args.checkpoint or [], labels):
#         m, t = load_finetuned(ckpt)
#         run(m, t, label)

#     if len(all_results) > 1:
#         print(f"\n{'Model':<35} {'Recovery':>10} ")
#         print("-" * 58)
#         for lbl, res in all_results.items():
#             print(f"{lbl:<35} {res['recovery']*100:>9.2f}% ")
#         print(f"{'ProteinMPNN':<35} {'52.40%':>10}")
#         print(f"{'ESM-IF':<35} {'51.60%':>10}")

#     os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
#     json.dump(all_results, open(args.out, "w"), indent=2)
#     print(f"\nSaved → {args.out}")


# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--checkpoint", nargs="+", default=None)
#     p.add_argument("--labels",     nargs="+", default=None)
#     p.add_argument("--zero_shot",  action="store_true")
#     p.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B")
#     p.add_argument("--test_file",  default="data/tokenized/test.jsonl")
#     p.add_argument("--out",        default="results/eval.json")
#     p.add_argument("--n_eval",     type=int, default=-1, help="-1 = all")
#     p.add_argument("--max_length", type=int, default=1024)
#     args = p.parse_args()
#     if not args.zero_shot and not args.checkpoint:
#         p.error("Provide --checkpoint and/or --zero_shot")
#     main(args)

"""
filteredinference.py
--------------
Evaluate LLM checkpoints on CATH inverse folding.

Usage:
    python filteredinference.py --checkpoint results/llm_3di --test_file data/tokenized3/test.jsonl
    python filteredinference.py --zero_shot --test_file data/tokenized3/test.jsonl
    python filteredinference.py \
        --checkpoint results/llm_3di_n500 results/llm_3di \
        --labels "n=500" "n=full" \
        --test_file data/tokenized3/test.jsonl
"""

import json, os, math, random, argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

STRUCT_BOS = "<struct>"
SEQ_BOS    = "<seq>"
AA_VOCAB   = set("ACDEFGHIKLMNPQRSTVWY")


def make_prompt(tokens_3di):
    """
    Must match training format exactly (filteredfinetuning_v2.py).
    ProstT5-style: space between every residue character, lower-case 3Di.
    Format: <struct> a c d e ... <seq>
    """
    spaced_3di = " ".join(list(tokens_3di.lower()))
    return f"{STRUCT_BOS} {spaced_3di} {SEQ_BOS}"


def setup_tokenizer(path):
    """
    Must match training tokenizer setup exactly.
    Only adds <struct> and <seq> special tokens — no individual AA/3Di chars.
    Individual letters are handled by spaces (ProstT5 approach).
    """
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": [STRUCT_BOS, SEQ_BOS]})
    return tok


def load_base(model_name, tokenizer):
    """Load base model, resize embeddings to match training (mean_resizing=True)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    return model.to(torch.bfloat16).to("cuda").eval()


def load_finetuned(checkpoint):
    cfg = PeftConfig.from_pretrained(checkpoint)
    # Tokenizer: checkpoint dir → parent → base model
    tok_path = next(
        (p for p in [checkpoint, os.path.dirname(checkpoint), cfg.base_model_name_or_path]
         if os.path.exists(os.path.join(p, "tokenizer_config.json"))),
        cfg.base_model_name_or_path,
    )
    tokenizer = setup_tokenizer(tok_path)
    model     = load_base(cfg.base_model_name_or_path, tokenizer)
    model     = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
    print(f"  ✓ LoRA merged from {checkpoint}")
    return model.eval(), tokenizer


def load_zeroshot(model_name):
    tokenizer = setup_tokenizer(model_name)
    model     = load_base(model_name, tokenizer)
    print(f"  ✓ Zero-shot {model_name}")
    return model, tokenizer


@torch.no_grad()
def compute_recovery(model, tokenizer, records, max_length):
    correct, total, examples = 0, 0, []

    for r in tqdm(records, desc="Recovery"):
        inputs = tokenizer(
            make_prompt(r["tokens_3di"]),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        out = model.generate(
            **inputs,
            # Each AA residue generates ~2 tokens (letter + space), +10 buffer
            max_new_tokens=r["length"] * 2 + 10,
            do_sample=False, #got 9% recovery
            temperature=None,
            top_p=None,
            # do_sample=True,
            # temperature=0.7,
            # top_p=0.9,   # reasonable nucleus size not explicitly stated in paper
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Model outputs spaced residues e.g. "A C D E F ..."
        # Collapse spaces first, then filter to valid AA chars
        pred = "".join(gen.split())
        pred = "".join(c for c in pred if c in AA_VOCAB)[:r["length"]]

        print(f"gen: {gen} \n pred: {pred}")
        ref = r["seq"]
        print(f"ref: {ref}")

        c = sum(p == q for p, q in zip(pred, ref))
        correct += c; total += len(ref)
        examples.append({
            "name": r["name"], "length": len(ref), "pred_len": len(pred),
            "ref": ref, "pred": pred, "recovery": c / len(ref),
        })
    return correct / total if total else 0.0, examples


def print_results(label, recovery, examples):
    def wavg(sub):
        return sum(e["recovery"]*e["length"] for e in sub) / sum(e["length"] for e in sub) \
               if sub else float("nan")
    short  = [e for e in examples if e["length"] <= 100]
    medium = [e for e in examples if 100 < e["length"] <= 300]
    long_  = [e for e in examples if e["length"] > 300]
    exact  = sum(1 for e in examples if e["pred_len"] == e["length"])

    print(f"\n{'='*55}\n  {label}  (n={len(examples)})\n{'='*55}")
    print(f"  Recovery (all)    : {recovery*100:6.2f}%")
    print(f"  Recovery (≤100aa) : {wavg(short)*100:6.2f}%  (n={len(short)})")
    print(f"  Recovery (101-300): {wavg(medium)*100:6.2f}%  (n={len(medium)})")
    print(f"  Recovery (>300aa) : {wavg(long_)*100:6.2f}%  (n={len(long_)})")
    print(f"  Exact length      : {exact}/{len(examples)}")
    print(f"  Baselines: ProteinMPNN=52.4%  ESM-IF=51.6%")
    for e in examples[:3]:
        flag = "✓" if e["pred_len"] == e["length"] else f"len={e['pred_len']}"
        print(f"\n  {e['name']} rec={e['recovery']:.3f} {flag}")
        print(f"    ref : {e['ref'][:60]}")
        print(f"    pred: {e['pred'][:60]}")


def main(args):
    random.seed(42)
    records = [json.loads(l) for l in open(args.test_file) if l.strip()]
    records = [r for r in records if r.get("tokens_3di") and r.get("seq")]
    if args.n_eval > 0:
        records = random.sample(records, min(args.n_eval, len(records)))
    print(f"Evaluating {len(records)} chains")

    all_results = {}

    def run(model, tokenizer, label):
        recovery, exs = compute_recovery(model, tokenizer, records, args.max_length)
        print_results(label, recovery, exs)
        all_results[label] = {"recovery": recovery, "n_eval": len(records), "examples": exs[:10]}
        del model; torch.cuda.empty_cache()

    if args.zero_shot:
        m, t = load_zeroshot(args.model_name)
        run(m, t, f"zero-shot ({args.model_name.split('/')[-1]})")

    labels = args.labels or args.checkpoint or []
    for ckpt, label in zip(args.checkpoint or [], labels):
        m, t = load_finetuned(ckpt)
        run(m, t, label)

    if len(all_results) > 1:
        print(f"\n{'Model':<35} {'Recovery':>10} ")
        print("-" * 58)
        for lbl, res in all_results.items():
            print(f"{lbl:<35} {res['recovery']*100:>9.2f}% ")
        print(f"{'ProteinMPNN':<35} {'52.40%':>10}")
        print(f"{'ESM-IF':<35} {'51.60%':>10}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(all_results, open(args.out, "w"), indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", nargs="+", default=None)
    p.add_argument("--labels",     nargs="+", default=None)
    p.add_argument("--zero_shot",  action="store_true")
    p.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--test_file",  default="data/tokenized/test.jsonl")
    p.add_argument("--out",        default="results/eval.json")
    p.add_argument("--n_eval",     type=int, default=-1, help="-1 = all")
    p.add_argument("--max_length", type=int, default=1024)
    args = p.parse_args()
    if not args.zero_shot and not args.checkpoint:
        p.error("Provide --checkpoint and/or --zero_shot")
    main(args)