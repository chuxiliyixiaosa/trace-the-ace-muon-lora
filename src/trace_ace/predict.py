from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Muon-LoRA sequence classifier")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features-output", type=Path)
    parser.add_argument("--max-length", type=int, default=10_240)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rows = []
    hidden_rows = []
    with args.jsonl.open("r", encoding="utf-8") as stream, torch.inference_mode():
        for line in stream:
            sample = json.loads(line)
            rendered = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            ).to(model.device)
            result = model(**inputs, output_hidden_states=args.features_output is not None)
            logits = result.logits.float().cpu().numpy()[0]
            probability = float(torch.softmax(torch.from_numpy(logits), dim=-1)[1])
            rows.append(
                {
                    "response_id": sample["response_id"],
                    "session_id": sample.get("session_id", ""),
                    "probability": probability,
                    "logit_0": float(logits[0]),
                    "logit_1": float(logits[1]),
                }
            )
            if args.features_output is not None:
                final_index = int(inputs["attention_mask"][0].sum().item()) - 1
                hidden_rows.append(result.hidden_states[-1][0, final_index].float().cpu().numpy())
    frame = pd.DataFrame(rows)
    if frame.response_id.duplicated().any() or not np.isfinite(frame.probability).all():
        raise RuntimeError("invalid prediction IDs or probabilities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    if args.features_output is not None:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.features_output,
            response_id=frame.response_id.to_numpy(dtype=str),
            hidden=np.asarray(hidden_rows, dtype=np.float32),
            logits=frame[["logit_0", "logit_1"]].to_numpy(dtype=np.float32),
        )


if __name__ == "__main__":
    main()
