#!/usr/bin/env python3
"""Prepare a deterministic ASearcher subset for framework-distillation rollout."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from typing import Any
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rows",
        type=int,
        default=500,
        help="Number of sampled rows, or -1 to keep every row remaining after exclusions.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--exclude-input",
        type=Path,
        action="append",
        default=[],
        help="Parquet whose questions must be excluded; may be repeated.",
    )
    parser.add_argument(
        "--prepare-script",
        type=Path,
        default=Path(__file__).parents[1] / "prepare.py",
    )
    return parser.parse_args()


def _as_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _question(value: Any) -> str:
    row = value.to_dict() if hasattr(value, "to_dict") else dict(value)
    for container_name in ("extra_info", "metadata"):
        container = _as_python(row.get(container_name))
        if isinstance(container, dict) and str(container.get("question") or "").strip():
            return str(container["question"])
    if str(row.get("question") or "").strip():
        return str(row["question"])
    prompt = _as_python(row.get("prompt"))
    if isinstance(prompt, (list, tuple)):
        for message in reversed(prompt):
            message = _as_python(message)
            if isinstance(message, dict) and str(message.get("role") or "").lower() == "user":
                return str(message.get("content") or "")
    messages = _as_python(row.get("messages"))
    if isinstance(messages, (list, tuple)):
        for message in reversed(messages):
            message = _as_python(message)
            if isinstance(message, dict) and str(message.get("role") or "").lower() == "user":
                return str(message.get("content") or "")
    return ""


def _question_key(value: Any) -> str:
    question = re.sub(r"\s+", " ", _question(value)).strip().casefold()
    return hashlib.sha256(question.encode("utf-8")).hexdigest() if question else ""


def main() -> None:
    args = parse_args()
    if args.rows == 0 or args.rows < -1:
        raise SystemExit("--rows must be positive or -1")
    source = pd.read_parquet(args.input)
    source_rows_before_exclusion = len(source)
    excluded_keys: set[str] = set()
    for exclude_path in args.exclude_input:
        excluded = pd.read_parquet(exclude_path)
        excluded_keys.update(
            key for _, row in excluded.iterrows() if (key := _question_key(row))
        )
    if excluded_keys:
        source_keys = source.apply(_question_key, axis=1)
        source = source.loc[~source_keys.isin(excluded_keys)].copy()
    excluded_source_rows = source_rows_before_exclusion - len(source)
    source_keys = source.apply(_question_key, axis=1)
    duplicate_mask = source_keys.ne("") & source_keys.duplicated(keep="first")
    deduplicated_source_rows = int(duplicate_mask.sum())
    if deduplicated_source_rows:
        source = source.loc[~duplicate_mask].copy()
    count = len(source) if args.rows == -1 else min(args.rows, len(source))
    if args.rows != -1 and count < args.rows:
        raise SystemExit(
            f"only {count} rows remain after exclusions, fewer than requested {args.rows}"
        )
    pilot = source.sample(n=count, random_state=args.seed, replace=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output.with_name(args.output.stem + ".raw.parquet")
    pilot.to_parquet(raw_output, index=False)
    subprocess.run(
        [
            sys.executable,
            str(args.prepare_script),
            "--input",
            str(raw_output),
            "--output",
            str(args.output),
            "--date", args.date,
            "--tool-prompt-mode",
            "native",
            "--native-tool-profile",
            "search_open_find",
        ],
        check=True,
    )
    print(
        f"Prepared {count} deterministic ASearcher rows (seed={args.seed}): "
        f"{args.output}; exclusion_keys={len(excluded_keys)} "
        f"excluded_source_rows={excluded_source_rows} "
        f"deduplicated_source_rows={deduplicated_source_rows}"
    )


if __name__ == "__main__":
    main()
