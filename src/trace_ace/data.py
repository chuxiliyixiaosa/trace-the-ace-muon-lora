from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from .headtail import transform_content


ID_TOKENS = ("response_id", "session_id", "learning_objective_id", "utterance_id")


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _remove_identifier_tokens(text: str) -> str:
    for token in ID_TOKENS:
        text = text.replace(token, token.replace("_", " "))
    return text


def load_transcript(path: Path) -> str:
    frame = pd.read_csv(path)
    if "utterance_id" in frame.columns:
        frame = frame.sort_values("utterance_id", kind="stable")
    turns = []
    for row in frame.itertuples(index=False):
        role = _text(getattr(row, "role"))
        content = _remove_identifier_tokens(_text(getattr(row, "content")))
        timestamp = _text(getattr(row, "timestamp")) if "timestamp" in frame.columns else ""
        turns.append(f"[{timestamp}] {role}: {content}" if timestamp else f"{role}: {content}")
    return "\n".join(turns)


def build_prompt(learning_objective: object, transcript: str) -> str:
    return "\n".join(
        [
            "You are given a tutoring dialogue from a K12 learning session.",
            "Decide whether the student's response for the target learning objective is correct.",
            "Return 1 if the response is correct, otherwise return 0.",
            "",
            f"Target learning objective: {_text(learning_objective)}",
            "",
            "Full dialogue:",
            transcript,
        ]
    )


def prepare_jsonl(
    features_path: Path,
    transcript_dir: Path,
    output_path: Path,
    labels_path: Path | None = None,
    tokenizer_path: str | None = None,
    headtail: bool = False,
) -> dict[str, int]:
    features = pd.read_csv(features_path, dtype={"response_id": str, "session_id": str})
    if labels_path is not None:
        labels = pd.read_csv(labels_path, dtype={"response_id": str})
        features = features.merge(labels[["response_id", "is_correct"]], on="response_id", validate="one_to_one")
    else:
        features["is_correct"] = 0

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if headtail else None
    cache: dict[str, str] = {}
    changed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in features.itertuples(index=False):
            session_id = _text(row.session_id)
            if session_id not in cache:
                cache[session_id] = load_transcript(transcript_dir / f"{session_id}.csv")
            prompt = build_prompt(getattr(row, "learning_objective", ""), cache[session_id])
            if headtail:
                prompt, was_changed = transform_content(tokenizer, prompt)
                changed += int(was_changed)
            sample = {
                "response_id": _text(row.response_id),
                "session_id": session_id,
                "messages": [{"role": "user", "content": prompt}],
                "label": int(row.is_correct),
            }
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return {"rows": len(features), "headtail_changed": changed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ms-swift JSONL from Trace the Ace files")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--headtail", action="store_true")
    args = parser.parse_args()
    if args.headtail and not args.model:
        parser.error("--model is required with --headtail")
    report = prepare_jsonl(
        args.features,
        args.transcripts,
        args.output,
        args.labels,
        args.model,
        args.headtail,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
