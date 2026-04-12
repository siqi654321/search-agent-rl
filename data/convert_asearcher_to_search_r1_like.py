#!/usr/bin/env python3
"""Convert ASearcher dataset to Search-R1-like parquet for VERL.

Target schema (per row):
- data_source: str (e.g. "searchR1_asearcher")
- prompt: list[{role, content}] (system + user)
- reward_model: dict with ground_truth (wrapped for token_f1)
- extra_info: dict with tools_kwargs (search tool create_kwargs)
- ability / metadata: preserved if present

Notes:
- The source dataset https://huggingface.co/datasets/aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample
  stores the real question/answer under `extra_info.question` and `extra_info.ground_truth`.
- token_f1 is enabled by setting reward_model.ground_truth = {"metric":"token_f1","target":...}
  (see `verl/utils/reward_score/search_r1_like_qa_em.py`).
"""

from __future__ import annotations

import argparse
import os
import random
from copy import deepcopy
from typing import Any, Optional

import pandas as pd
from datasets import load_dataset


DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless assistant."
DEFAULT_USER_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)


def _safe_dict(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _normalize_target(x: Any) -> Any:
    # Keep list as-is, otherwise coerce to string.
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x if i is not None]
    return str(x)


def build_user_content(question: str, prefix: Optional[str], template: Optional[str]) -> str:
    q = (question or "").strip()
    if template:
        return str(template).format(question=q)
    return (prefix or "").rstrip("\n") + q


def convert_row(
    row: dict,
    split_name: str,
    idx: int,
    *,
    system_prompt: str,
    user_prefix: str,
    user_template: Optional[str],
    data_source: str,
    force_token_f1: bool,
) -> dict:
    row = deepcopy(row)
    extra = _safe_dict(row.get("extra_info"))

    question = str(extra.get("question") or "").strip()
    gt = extra.get("ground_truth")
    target = _normalize_target(gt)

    # Prompts
    user_content = build_user_content(question, user_prefix, user_template)
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Reward model ground truth
    reward_model = _safe_dict(row.get("reward_model"))
    reward_model = deepcopy(reward_model)
    if force_token_f1:
        reward_model["ground_truth"] = {"metric": "token_f1", "target": target}
    else:
        reward_model["ground_truth"] = target

    # Tool kwargs (SearchTool expects this structure in Search-R1-like datasets)
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": reward_model["ground_truth"],
                "question": question,
                "data_source": data_source,
            }
        }
    }

    # Extra info (keep source fields too)
    extra_info = deepcopy(extra)
    extra_info.update(
        {
            "index": extra_info.get("id", idx),
            "need_tools_kwargs": True,
            "question": question,
            "split": split_name,
            "tools_kwargs": tools_kwargs,
        }
    )

    out = {
        "data_source": data_source,
        "prompt": prompt,
        "ability": row.get("ability"),
        "reward_model": reward_model,
        "extra_info": extra_info,
        "metadata": row.get("metadata"),
    }
    return out


def _split_train_test(records: list[dict], test_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if test_ratio <= 0:
        return records, []
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    n_test = max(1, int(len(records) * test_ratio))
    test_set = set(idxs[:n_test])
    train, test = [], []
    for i, r in enumerate(records):
        (test if i in test_set else train).append(r)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ASearcher to Search-R1-like parquet.")
    parser.add_argument(
        "--hf_repo_id",
        default="aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample",
        help="HuggingFace dataset repo id",
    )
    parser.add_argument("--output_dir", required=True, help="Output dir to write train.parquet/test.parquet")
    parser.add_argument(
        "--data_source",
        default="searchR1_openresearcher",
        help=(
            "Output data_source string. 为了不改 VERL 代码，请使用内置支持的 Search-R1-like data_source，"
            "例如 searchR1_openresearcher/searchR1_nq/searchR1_hotpotqa 等。"
        ),
    )
    parser.add_argument("--test_ratio", type=float, default=0.02, help="Create test split from train if missing")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user_prompt_prefix", default=DEFAULT_USER_PREFIX)
    parser.add_argument(
        "--user_prompt_template",
        default=None,
        help="Optional template using '{question}', overrides prefix if set",
    )
    parser.add_argument(
        "--token_f1",
        action="store_true",
        help="Enable token_f1 by wrapping reward_model.ground_truth with metric/target",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print a few converted samples and output schema; do not write parquet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Only convert first N samples (useful with --dry_run).",
    )
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset(args.hf_repo_id)
    # Most likely only train split exists.
    train_ds = ds.get("train") if isinstance(ds, dict) else ds
    if train_ds is None:
        raise ValueError("dataset does not contain a train split")

    total = len(train_ds)
    limit = args.limit if args.limit and args.limit > 0 else total
    limit = min(limit, total)
    records = [train_ds[i] for i in range(limit)]
    converted = [
        convert_row(
            r,
            split_name="train",
            idx=i,
            system_prompt=args.system_prompt,
            user_prefix=args.user_prompt_prefix,
            user_template=args.user_prompt_template,
            data_source=args.data_source,
            force_token_f1=args.token_f1,
        )
        for i, r in enumerate(records)
    ]

    train_rows, test_rows = _split_train_test(converted, test_ratio=args.test_ratio, seed=args.seed)

    # Update split label in extra_info for test
    if test_rows:
        for r in test_rows:
            ei = r.get("extra_info")
            if isinstance(ei, dict):
                ei["split"] = "test"

    if args.dry_run:
        print(f"[dry-run] loaded={total} converted={len(converted)} train={len(train_rows)} test={len(test_rows)}")
        if train_rows:
            sample = train_rows[0]
            print("[dry-run] columns:", list(sample.keys()))
            print("[dry-run] sample.data_source:", sample.get("data_source"))
            print("[dry-run] sample.prompt:", sample.get("prompt"))
            rm = sample.get("reward_model")
            if isinstance(rm, dict):
                print("[dry-run] sample.reward_model.ground_truth:", rm.get("ground_truth"))
            ei = sample.get("extra_info")
            if isinstance(ei, dict):
                print("[dry-run] sample.extra_info.question:", ei.get("question"))
                print("[dry-run] sample.extra_info.tools_kwargs keys:", list(_safe_dict(ei.get("tools_kwargs")).keys()))
        return

    pd.DataFrame(train_rows).to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    if test_rows:
        pd.DataFrame(test_rows).to_parquet(os.path.join(out_dir, "test.parquet"), index=False)

    print(f"Wrote train={len(train_rows)} test={len(test_rows)} to {out_dir}")


if __name__ == "__main__":
    main()

