#!/usr/bin/env python3
"""Uniform sampling, dual-route runtime audit and final length-evidence selection."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from . import analyze as analyzer
except ImportError:  # Direct python3 data_code/... invocation.
    import analyze as analyzer




def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_excluded_questions(paths: list[Path] | tuple[Path, ...], source_digest: str) -> tuple[set[str], list[dict]]:
    """Use complete frozen samples, not their post-rollout selected subsets."""
    excluded: set[str] = set()
    inputs = []
    for path in sorted({p.resolve() for p in paths}):
        payload = path.read_bytes()
        report = json.loads(payload)
        if not isinstance(report, dict) or report.get("schema") not in {
            "browsecomp_safe_uniform_batch_sample_v1", "browsecomp_safe_uniform_batch_sample_v2"
        }:
            raise ValueError(f"Expected a complete batch sample.report.json for exclusion: {path}")
        if report.get("source_sha256") != source_digest:
            raise ValueError(f"Exclusion sample has a different source SHA256: {path}")
        ids = report.get("query_ids")
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(qid, str) or not qid.strip() or qid != qid.strip() for qid in ids)
                or len(set(ids)) != len(ids)
                or type(report.get("rows")) is not int or report["rows"] != len(ids)):
            raise ValueError(f"Exclusion sample must contain rows distinct, nonempty query_ids: {path}")
        excluded.update(ids)
        inputs.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(ids)})
    return excluded, inputs


def prepare_sample(source: Path, stage: Path, rows: int, seed: int,
                   exclude_sample_reports: list[Path] | tuple[Path, ...] = ()) -> dict:
    """Hash-shuffle IDs without looking at question difficulty or rewards."""
    output, report_path = stage / "questions.parquet", stage / "sample.report.json"
    identity = {"source": str(source.resolve()), "source_sha256": sha256(source),
                "rows": rows, "sample_seed": seed}
    excluded, exclusion_inputs = _load_excluded_questions(exclude_sample_reports, identity["source_sha256"])
    if output.exists() or report_path.exists():
        if not output.is_file() or not report_path.is_file():
            raise ValueError("Sample cache differs or is incomplete; choose a new --stage-dir (no files overwritten)")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (any(report.get(k) != v for k, v in identity.items())
                or report.get("exclude_sample_reports", []) != exclusion_inputs
                or sha256(output) != report["sample_sha256"]):
            raise ValueError("Sample cache differs; choose a new --stage-dir (no files overwritten)")
        return report
    table = pq.read_table(source)
    ids = [str(v).strip() if v is not None else "" for v in table.column("query_id").to_pylist()]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Source query_id must be nonempty and unique")
    unknown = excluded - set(ids)
    if unknown:
        raise ValueError(f"Exclusion sample contains query_ids absent from source: {sorted(unknown)[:5]}")
    eligible = [i for i, qid in enumerate(ids) if qid not in excluded]
    if not 0 < rows <= len(eligible):
        raise ValueError(f"Requested {rows} questions from only {len(eligible)} remaining source rows "
                         f"after excluding {len(excluded)}")
    indices = sorted(eligible, key=lambda i: hashlib.sha256(f"{seed}\0{ids[i]}".encode()).digest())[:rows]
    overlap = excluded.intersection(ids[i] for i in indices)
    if overlap:
        raise ValueError("New sample overlaps excluded questions; no files written")
    stage.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(pa.array(indices, type=pa.int64())), output, compression="zstd")
    report = {"schema": "browsecomp_safe_uniform_batch_sample_v2", **identity,
              "source_rows": len(ids), "sample_sha256": sha256(output),
              "query_ids": [ids[i] for i in indices], "source_row_indices": indices,
              "exclude_sample_reports": exclusion_inputs, "excluded_questions": len(excluded),
              "available_rows_after_exclusion": len(eligible), "remaining_rows_after_sample": len(eligible) - rows,
              "sample_excluded_overlap": len(overlap),
              "policy": "uniform hash shuffle without replacement; no reward/behavior prescreen; source untouched"}
    write_json(report_path, report)
    return report




def audit_rollouts(path: Path, expected_questions: int, rollout_n: int, min_applied_rate: float) -> dict:
    groups: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        counts["rows"] += 1
        key = row.get("query_id")
        if key is None or str(key) == "":
            key = row.get("question")
        if key is None or str(key) == "":
            raise ValueError(f"missing question identity at {path}:{line_number}")
        groups[str(key)] += 1
        if "search_calls" not in row:
            counts["rows_missing_search_calls"] += 1
        counts["search_calls"] += int(row.get("search_calls") or 0)
        events = row.get("search_frontier_events") or []
        if not isinstance(events, list):
            raise ValueError(f"search_frontier_events is not a list at line {line_number}")
        for event in events:
            counts["search_events"] += 1
            if "dual_retrieval_applied" not in event:
                counts["events_missing_dual_telemetry"] += 1
            counts["dual_enabled"] += event.get("dual_retrieval_enabled") is True
            counts["dual_applied"] += event.get("dual_retrieval_applied") is True
            error = str(event.get("dual_retrieval_fallback_error") or "").strip()
            if error:
                counts["fallback_errors"] += 1
                errors[error] += 1
    # Use all search events as the denominator, including missing route fields.
    applied_rate = counts["dual_applied"] / max(counts["search_events"], 1)
    coverage = counts["search_events"] / max(counts["search_calls"], 1)
    failures = []
    if len(groups) != expected_questions or any(n != rollout_n for n in groups.values()):
        failures.append(f"expected {expected_questions} questions x {rollout_n} branches")
    if counts["rows_missing_search_calls"] or counts["events_missing_dual_telemetry"]:
        failures.append("missing search/dual telemetry; do not infer the route from startup logs")
    if not counts["search_calls"] or not 0.95 <= coverage <= 1.0:
        failures.append(f"search event coverage={coverage:.4f}; expected 0.95..1.0")
    if applied_rate < min_applied_rate:
        failures.append(f"dual applied rate={applied_rate:.4f} < {min_applied_rate}")
    if counts["fallback_errors"]:
        failures.append("AgentIR fallback errors occurred; inspect them before accepting this comparison")
    return {
        "schema": "browsecomp_dual_route_rollout_audit_v1",
        "rollouts": str(path),
        "counts": {name: counts[name] for name in (
            "rows", "search_calls", "search_events", "dual_enabled", "dual_applied",
            "fallback_errors", "rows_missing_search_calls", "events_missing_dual_telemetry",
        )},
        "questions": len(groups),
        "group_size_counts": dict(sorted(Counter(groups.values()).items())),
        "dual_applied_rate": applied_rate,
        "search_event_coverage": coverage,
        "minimum_dual_applied_rate": min_applied_rate,
        "fallback_error_examples": dict(errors.most_common(5)),
        "passed": not failures,
        "failures": failures,
        "policy": "runtime acceptance only; no trajectory is removed or changed",
    }


def is_length_evidence_core(row: dict) -> bool:
    return bool(row["rollouts"] == 8 and row["exact_telemetry"] and row["informative"]
                and row["max_base_f1"] >= 1 - 1e-12
                and row["best_evidence_supported"] and row["best_length_matched"])


def analyze_batch(stage: Path, reward_path: Path) -> dict:
    sample = json.loads((stage / "sample.report.json").read_text())
    dump = stage / "rollouts-n8/0.jsonl"
    audit = audit_rollouts(dump, sample["rows"], 8, 0.95)
    write_json(stage / "dual.rollout-audit.json", audit)
    if not audit["passed"]:
        raise ValueError(f"Dual-route runtime audit failed; raw outputs retained: {audit['failures']}")
    os.environ["SEARCH_STRICT_EVIDENCE_COEF_ENABLED"] = "0"
    reward = analyzer._load_reward(reward_path)
    observed = Counter()
    bundle_ids = {}
    reward_sum = 0.0
    with dump.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            observed[str(row["query_id"])] += 1
            ground_truth = analyzer._mapping(row.get("gts"))
            if not ground_truth.get("answer_docid"):
                raise ValueError("V4 row lacks designated answer_docid; do not use the BC-reference fallback")
            bundle_ids[str(row["query_id"])] = str(ground_truth.get("chain_bundle_id") or "")
            score = float(reward.strict_terminal_reward_details(row["output"], analyzer._targets(row["gts"]))["base_f1"])
            if not abs(float(row["score"]) - score) <= 1e-6:
                raise ValueError(f"Stored/strict reward mismatch: {row['query_id']}")
            reward_sum += score
    if set(observed) != set(sample["query_ids"]):
        raise ValueError("Rollout identities do not match the frozen random sample")
    out = stage / "analysis"
    args = analyzer.build_parser().parse_args([
        "--rollouts", str(dump), "--reward-script", str(reward_path),
        "--selection-mode", "success_profile", "--best-evidence-mode", "answer_doc_and_target",
        "--require-exact-telemetry", "--expected-rollouts-per-question", "8", "--min-f1-range", "1e-6",
        "--min-best-tier-mean-turns", "60", "--max-best-tier-mean-turns", "105",
        "--min-best-tier-mean-response-tokens", "40000", "--max-best-tier-mean-response-tokens", "70000",
        "--target-selected-groups", "400",
        "--report", str(out / "full-profile.report.json"), "--details-jsonl", str(out / "details.jsonl"),
        "--selected-jsonl", str(out / "full-profile.selected.jsonl"),
    ])
    print("[BCP-BATCH] Full success-profile report is diagnostic only; core selection follows separately.", flush=True)
    full_report = analyzer.run(args)
    details = [json.loads(line) for line in (out / "details.jsonl").read_text().splitlines() if line.strip()]
    if not all(row["exact_telemetry"] for row in details):
        raise ValueError("Missing exact turn/token telemetry; no core training parquet is materialized")
    selected = [dict(row, full_success_profile_matched=row["selected_behavior_matched"],
                     selected_behavior_matched=True, selection_mode="length_evidence_core")
                for row in details if is_length_evidence_core(row)]
    ids = {row["group_id"].removeprefix("query_id:") for row in selected}
    table = pq.read_table(stage / "questions.parquet")
    if sha256(stage / "questions.parquet") != sample["sample_sha256"]:
        raise ValueError("Frozen sample has changed since rollout")
    indices = [i for i, qid in enumerate(table.column("query_id").to_pylist()) if str(qid) in ids]
    if len(indices) != len(ids):
        raise ValueError("Selected identities cannot be joined to the original sample")
    pq.write_table(table.take(pa.array(indices, type=pa.int64())), out / "length-evidence-core.parquet", compression="zstd")
    (out / "length-evidence-core.selected.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    timing = json.loads((stage / "wall.report.json").read_text())
    seconds = timing["rollout_wall_seconds"]
    report = {
        "schema": "browsecomp_safe_uniform_batch_yield_v1", "questions": sample["rows"],
        "rollout_rows": sum(observed.values()), "rollout_wall_seconds": seconds,
        "rollout_wall_hours": seconds / 3600, "trajectories_per_hour": sum(observed.values()) * 3600 / seconds if seconds else None,
        "mean_strict_reward": reward_sum / sum(observed.values()), "dual_applied_rate": audit["dual_applied_rate"],
        "counts": {"raw_f1_varying": full_report["counts"].get("raw_f1_varying", 0),
                   "valid_f1_varying": full_report["counts"].get("valid_f1_varying", 0),
                   "length_evidence_core": len(selected),
                   "full_success_profile": full_report["counts"].get("selected_behavior_matched", 0)},
        "length_evidence_core_rate": len(selected) / sample["rows"],
        "selected_unique_known_chain_bundles": len({bundle_ids[q] for q in ids if bundle_ids[q]}),
        "selected_questions_missing_bundle_id": sum(not bundle_ids[q] for q in ids),
        "policy": {"selection": "valid F1 variance >=1e-6 + exact success + best-tier body support + best-tier length",
                   "best_tier_turns": [60, 105], "best_tier_response_tokens": [40000, 70000],
                   "search_ratio_and_success_failure_gap": "diagnostic only, not core gates",
                   "evidence": "same best trajectory inspected answer_docid and target occurred in some inspection; not a causal proof",
                   "capacity_is_not_a_gate": True, "source_rows_preserved": True,
                   "sample_seed_is_not_generation_seed": True},
        "outputs": {"parquet": str(out / "length-evidence-core.parquet"), "selected_jsonl": str(out / "length-evidence-core.selected.jsonl")},
    }
    write_json(out / "summary.json", report)
    print("[BCP-BATCH] SUMMARY " + json.dumps(report, ensure_ascii=False), flush=True)
    return report
