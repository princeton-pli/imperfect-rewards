import argparse
import json
import random
import re
from datasets import load_dataset
from pathlib import Path
import shlex
import sys
import os
import itertools
from transformers import AutoTokenizer
import instructions_registry

FORMAT_TEMPLATES = {
    "gsm8k": {
        "template": "#### {value}",
        "pattern": r"#### (\-?[0-9\.\,]+)",
    }
}

def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]

def extract_regular_gold_math(answer_text: str) -> str:
    return remove_boxed(last_boxed_only_string(answer_text))

def extract_regular_gold(answer_text: str, gold_re, ) -> str:
    matches = re.findall(gold_re, answer_text)
    if not matches:
        raise ValueError("Could not find gold answer gold_re pattern.")
    return matches[-1].strip()

def eligible_instruction_ids(blocklist=None):
    blocklist = set(blocklist or [])
    eligible = []
    for instr_id, cls in instructions_registry.INSTRUCTION_DICT.items():
        if instr_id in blocklist:
            continue
        eligible.append(instr_id)
    return eligible

def build_prompt(question: str, constraint_texts: list[str], instr_ids: list[str]) -> str:
    constraint_str = build_constraint_only(constraint_texts, instr_ids)
    return f"{question.strip()} {constraint_str}"

def build_constraint_only(constraint_texts: list[str], instr_ids: list[str]) -> str:
    constraints_block = " ".join(
        t for i, t in enumerate(constraint_texts)
    )
    return f"{constraints_block}"

def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def filter_by_length(args, tokenizer, example):
    prompt_not_too_long = False
    if args.max_prompt_length > 0:
        prompt_length = len(tokenizer.encode(example["messages"][0]["content"]))
        prompt_not_too_long = prompt_length <= args.max_prompt_length

    return prompt_not_too_long

def _load_allowlist(allowlist_arg: str | None) -> list[str]:
    if not allowlist_arg:
        return []

    allowlist_path = Path(allowlist_arg)
    if allowlist_path.exists() and allowlist_path.is_file():
        with allowlist_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict):
            if "allowlist" in data and isinstance(data["allowlist"], list):
                return [str(x).strip() for x in data["allowlist"] if str(x).strip()]
            return [str(x).strip() for x in data.keys() if str(x).strip()]
        raise ValueError("Allowlist JSON must be a list or dict.")

    return [x.strip() for x in allowlist_arg.split(",") if x.strip()]

def _sanitize_constraint_id(instr_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", instr_id)

def normalize_constraint_sets(constraint_sets):
    normalized = []
    for i, item in enumerate(constraint_sets):
        if isinstance(item, str):
            normalized.append([item])
        elif isinstance(item, (list, tuple)):
            normalized.append(list(item))
    return normalized

def _write_dataset(
    args,
    ds,
    dataset_name_constraint: str,
    out_dir: Path,
    pool: list[str],
    overrides: dict,
    tokenizer,
    constraint_sets: list[list[str]] | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = " ".join([shlex.quote(x) for x in sys.argv])
    safe_write_text(out_dir / "build_command.txt", cmd + "\n")
    safe_write_text(out_dir / "build_args.json", json.dumps(vars(args), indent=2) + "\n")

    jsonl_path = out_dir / "dataset.jsonl"
    total_rows = 0
    keyword = "prompt" # For HuggingFaceH4/ultrafeedback_binarized
    gold = None # No gold answer for HuggingFaceH4/ultrafeedback_binarized
    constraint_sets = constraint_sets or []
    use_constraint_sets = len(constraint_sets) > 0

    with open(jsonl_path, "w") as f:
        for i, ex in enumerate(ds):
            instr_id_sets = constraint_sets if use_constraint_sets else [random.sample(pool, k=min(args.k, len(pool)))]
            rows_to_write = []
            for instr_ids in instr_id_sets:
                kwargs_list = []
                constraint_texts = []
                for instr_id in instr_ids:
                    override_raw = overrides.get(instr_id, {})
                    cls = instructions_registry.INSTRUCTION_DICT[instr_id]
                    inst = cls(instr_id)
                    desc = inst.build_description(**override_raw)

                    kwargs_list.append(override_raw)
                    constraint_texts.append(desc)

                ifeval_label_str = repr([{
                    "instruction_id": instr_ids,
                    "kwargs": kwargs_list,
                }])                  
                
                prompt = build_prompt(ex[keyword].strip(), constraint_texts, instr_ids)
                if args.add_step_by_step_to_user:
                    prompt = "Let's think step by step.\n" + prompt

                rows_to_write.append({
                    "key": f"ds_{args.split_to_use}_{i}",
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }],
                    "dataset": [dataset_name_constraint, "ifeval"] if dataset_name_constraint else ["ifeval"],
                    "ground_truth": [gold, ifeval_label_str] if gold else [ifeval_label_str],
                    "constraint": build_constraint_only(constraint_texts, instr_ids),
                    "instruction_id_list": instr_ids,
                })

            for row in rows_to_write:
                if args.max_prompt_length > 0:
                    if filter_by_length(args, tokenizer, row):
                        f.write(json.dumps(row) + "\n")
                        total_rows += 1
                else:
                    f.write(json.dumps(row) + "\n")
                    total_rows += 1

    summary = {
        "num_initial_examples": len(ds),
        "num_final_examples": total_rows,
        "split_to_use": args.split_to_use,
        "k_constraints": args.k,
        "seed": args.seed,
        "max_prompt_length": args.max_prompt_length,
        "tokenizer_name": args.tokenizer_name,
        "add_step_by_step_to_user": args.add_step_by_step_to_user,
    }
    if use_constraint_sets:
        _ = normalize_constraint_sets(constraint_sets)
        if dataset_name_constraint is not None:
            _.append([dataset_name_constraint])
        summary["constraint_sets"] = list(itertools.chain.from_iterable(_))
    safe_write_text(out_dir / "build_summary.json", json.dumps(summary, indent=2) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="openai/gsm8k", help="HuggingFace dataset name")
    ap.add_argument("--out_dir", required=True, help="Output dir")
    ap.add_argument("--max_prompt_length", type=int, default=512,
                    help="If >0, filter out examples with prompt longer than this (in tokens).")
    ap.add_argument("--tokenizer_name", type=str, default=None,
                    help="Tokenizer name for length filtering (HuggingFace). Required if max_prompt_length > 0")
    ap.add_argument("--add_step_by_step_to_user", action="store_true", 
                    help="If set, appends 'Let's think step by step.' to the user prompt.")
    ap.add_argument("--split_to_use", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=2, help="Constraints per example (also controls combination size when --create_pairs is set; k=2 → pairs, k=3 → triplets, etc.)")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--allowlist", type=str, default=None,
                    help="Comma-separated instruction_ids or JSON file path (optional)")
    ap.add_argument("--create_pairs", action="store_true",
                    help="If set, create all unique k-tuples from allowlist and apply each as constraints.")
    ap.add_argument("--instruction_overrides", type = str, default=None,
                    help="Path to JSON file with instruction_id to args dict overrides (optional)") 
    ap.add_argument("--blocklist", type=str, default="format:no_whitespace",
                    help="Comma-separated instruction_ids to exclude (optional)")
    args = ap.parse_args()

    random.seed(args.seed)

    if args.dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")[args.split_to_use]
        dataset_name_constraint = None
    else:
        raise NotImplementedError("Currently only UltraFeedback is supported")

    ds = ds.shuffle(seed=args.seed)
    if args.max_examples is not None:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    tokenizer = None
    if args.max_prompt_length > 0 and args.tokenizer_name is None:
        raise ValueError("--tokenizer_name is required when --max_prompt_length > 0.")
    if args.max_prompt_length > 0 and args.tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    pool, blocklist, overrides = [], [], {}
    if instructions_registry is None:
        raise RuntimeError("instructions_registry could not be imported. Install/enable IFBench to use constraints.")
    blocklist = [x.strip() for x in args.blocklist.split(",") if x.strip()] if args.blocklist else []

    if args.allowlist:
        pool = _load_allowlist(args.allowlist)
    else:
        pool = eligible_instruction_ids(blocklist=blocklist)

    if len(pool) == 0:
        raise RuntimeError("No eligible IFBench instruction_ids found. Provide --allowlist explicitly")

    overrides = {}
    if args.instruction_overrides:
        with open(args.instruction_overrides, 'r') as f:
            overrides = json.load(f)

    base_out_dir = Path(args.out_dir)
    if args.create_pairs:
        if len(pool) < args.k:
            raise RuntimeError(f"--create_pairs requires at least {args.k} allowlist items.")
        base_out_dir.mkdir(parents=True, exist_ok=True)
        for pair in itertools.combinations(pool, args.k):
            pair_ids = list(pair)
            pair_dir = base_out_dir / f"{'__'.join(_sanitize_constraint_id(pid) for pid in pair_ids)}__{args.split_to_use}"
            _write_dataset(
                args=args,
                ds=ds,
                dataset_name_constraint=dataset_name_constraint,
                out_dir=pair_dir,
                pool=pair_ids,
                overrides=overrides,
                tokenizer=tokenizer,
                constraint_sets=[pair_ids],
            )

if __name__ == "__main__":
    main()