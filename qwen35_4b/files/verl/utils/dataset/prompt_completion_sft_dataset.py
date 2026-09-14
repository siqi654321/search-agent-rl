"""SFT dataset for already-rendered prompts and corrected completions.

Unlike MultiTurnSFTDataset this class does not re-apply a chat template.  It is
intended for framework-distillation traces whose ``context_text`` is decoded
directly from the rollout's prompt token IDs.  Only ``chosen_text`` receives
loss; the complete rendered context is masked.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset

from verl.utils.fs import copy_local_path_from_hdfs


class PromptCompletionSFTDataset(Dataset):
    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer,
        config,
        processor: Optional[object] = None,
        max_samples: int = -1,
    ):
        # This dataset consumes already-rendered text and tokenizes it only with
        # ``tokenizer`` below.  Transformers 5.x may return a non-None
        # AutoProcessor even for text-only Qwen3.5 checkpoints, so processor
        # presence is not a reliable multimodal signal.  It is intentionally
        # ignored; no image/video fields are accepted or processed here.
        del processor
        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        local_paths = [copy_local_path_from_hdfs(path, verbose=True) for path in parquet_files]
        self.dataframe = pd.concat(
            [pd.read_parquet(path, dtype_backend="pyarrow") for path in local_paths],
            ignore_index=True,
        )
        self.tokenizer = tokenizer
        self.context_key = config.get("context_key", "context_text")
        self.completion_key = config.get("completion_key", "chosen_text")
        self.loss_spans_key = config.get("loss_spans_key")
        self.max_length = int(config.get("max_length", 49152))
        self.truncation = str(config.get("truncation", "error"))
        self.pad_mode = str(config.get("pad_mode", "no_padding"))
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = config.get("seed")
        if self.pad_mode not in {"right", "no_padding"}:
            raise ValueError("pad_mode must be 'right' or 'no_padding'")
        if self.truncation not in {"error", "left", "right"}:
            raise ValueError("truncation must be 'error', 'left', or 'right'")
        if max_samples > 0 and max_samples < len(self.dataframe):
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(len(self.dataframe), size=max_samples, replace=False)
            else:
                indices = np.arange(max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def _parse_loss_spans(raw_spans, *, completion_length: int, index: int) -> list[tuple[int, int]]:
        """Parse fail-closed character spans relative to ``chosen_text``.

        The parquet stores spans as JSON to avoid pandas/pyarrow version-specific
        nested-list coercions.  Spans must be sorted, non-overlapping, non-empty,
        and fully contained in the completion.
        """

        if isinstance(raw_spans, str):
            try:
                raw_spans = json.loads(raw_spans)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid loss-span JSON at row {index}: {exc}") from exc
        if not isinstance(raw_spans, (list, tuple, np.ndarray)):
            raise ValueError(f"loss spans at row {index} must be a JSON/list sequence")

        spans: list[tuple[int, int]] = []
        previous_end = 0
        for raw_span in raw_spans:
            if not isinstance(raw_span, (list, tuple, np.ndarray)) or len(raw_span) != 2:
                raise ValueError(f"invalid loss span at row {index}: {raw_span!r}")
            try:
                start, end = int(raw_span[0]), int(raw_span[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-integer loss span at row {index}: {raw_span!r}") from exc
            if start < previous_end or start < 0 or end <= start or end > completion_length:
                raise ValueError(
                    f"out-of-order/out-of-range loss span at row {index}: "
                    f"{(start, end)!r} for completion length {completion_length}"
                )
            spans.append((start, end))
            previous_end = end
        if not spans:
            raise ValueError(f"empty loss spans at row {index}")
        return spans

    def _encode_completion(self, completion: str, row, *, index: int) -> tuple[list[int], list[int]]:
        """Tokenize a completion and optionally mask it to explicit char spans."""

        if not self.loss_spans_key:
            completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
            return completion_ids, [1] * len(completion_ids)

        if self.loss_spans_key not in row.index:
            raise ValueError(
                f"configured loss_spans_key={self.loss_spans_key!r} is absent at row {index}"
            )
        spans = self._parse_loss_spans(
            row[self.loss_spans_key], completion_length=len(completion), index=index
        )
        try:
            encoded = self.tokenizer(
                completion,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except (NotImplementedError, TypeError) as exc:
            raise ValueError(
                "explicit completion loss spans require a fast tokenizer with "
                "return_offsets_mapping support"
            ) from exc
        completion_ids = list(encoded["input_ids"])
        offsets = list(encoded.get("offset_mapping") or [])
        if len(offsets) != len(completion_ids):
            raise ValueError(
                f"token offset count mismatch at row {index}: "
                f"{len(offsets)} offsets for {len(completion_ids)} tokens"
            )

        # A token receives loss when its source-character interval overlaps a
        # requested span.  Added tokens written explicitly in the completion
        # (for example <|im_end|>) have real offsets in Qwen's fast tokenizer.
        completion_mask: list[int] = []
        for raw_offset in offsets:
            if not isinstance(raw_offset, (list, tuple, np.ndarray)) or len(raw_offset) != 2:
                raise ValueError(f"invalid tokenizer offset at row {index}: {raw_offset!r}")
            token_start, token_end = int(raw_offset[0]), int(raw_offset[1])
            completion_mask.append(
                int(
                    token_end > token_start
                    and any(
                        token_start < span_end and token_end > span_start
                        for span_start, span_end in spans
                    )
                )
            )
        if not any(completion_mask):
            raise ValueError(f"loss spans at row {index} did not select any completion token")
        return completion_ids, completion_mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        context = str(row[self.context_key])
        completion = str(row[self.completion_key])
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        completion_ids, completion_mask = self._encode_completion(completion, row, index=index)
        if not completion_ids:
            raise ValueError(f"empty completion at row {index}")
        input_ids = context_ids + completion_ids
        loss_mask = [0] * len(context_ids) + completion_mask

        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            if self.truncation == "error":
                raise ValueError(
                    f"sequence at row {index} has {len(input_ids)} tokens, exceeding {self.max_length}"
                )
            if self.truncation == "left":
                if overflow >= len(context_ids):
                    raise ValueError(
                        f"left truncation at row {index} would remove the correction target"
                    )
                input_ids = input_ids[overflow:]
                loss_mask = loss_mask[overflow:]
            else:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                if not any(loss_mask):
                    raise ValueError(f"right truncation at row {index} removed the correction target")

        attention_mask = [1] * len(input_ids)
        if self.pad_mode == "right" and len(input_ids) < self.max_length:
            padding = self.max_length - len(input_ids)
            pad_token_id = self.tokenizer.pad_token_id or 0
            input_ids += [pad_token_id] * padding
            loss_mask += [0] * padding
            attention_mask += [0] * padding

        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        result = {
            "input_ids": input_tensor,
            "position_ids": torch.arange(len(input_ids), dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }
        if self.pad_mode == "right":
            result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        return result
