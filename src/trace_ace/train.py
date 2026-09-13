from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def build_command(args: argparse.Namespace) -> list[str]:
    plugin = Path(__file__).with_name("muon_audit.py").resolve()
    return [
        args.swift,
        "sft",
        "--model",
        args.model,
        "--model_type",
        "qwen3",
        "--model_revision",
        "master",
        "--template",
        "qwen3_nothinking",
        "--torch_dtype",
        "bfloat16",
        "--tuner_type",
        "lora",
        "--task_type",
        "seq_cls",
        "--problem_type",
        "single_label_classification",
        "--num_labels",
        "2",
        "--max_length",
        "10240",
        "--truncation_strategy",
        "right",
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "2.5e-5",
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        "0.03",
        "--weight_decay",
        "0.1",
        "--max_grad_norm",
        "1.0",
        "--seed",
        "42",
        "--data_seed",
        "42",
        "--lora_rank",
        "256",
        "--lora_alpha",
        "512",
        "--lora_dropout",
        "0.05",
        "--target_modules",
        "all-linear",
        "--gradient_checkpointing",
        "true",
        "--attn_impl",
        "sdpa",
        "--dataset_num_proc",
        str(args.workers),
        "--dataloader_num_workers",
        str(args.workers),
        "--packing",
        "false",
        "--padding_free",
        "false",
        "--label_smoothing_factor",
        "0",
        "--save_only_model",
        "true",
        "--save_strategy",
        "epoch",
        "--save_total_limit",
        "1",
        "--logging_steps",
        "100",
        "--split_dataset_ratio",
        "0",
        "--adam_beta1",
        "0.9",
        "--adam_beta2",
        "0.95",
        "--optim",
        "adamw_torch_fused",
        "--dataset",
        str(args.dataset.resolve()),
        "--output_dir",
        str(args.output.resolve()),
        "--optimizer",
        "muon",
        "--local_repo_path",
        str(args.muon_repo.resolve()),
        "--external_plugins",
        str(plugin),
        "--callbacks",
        "muon_audit",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the audited four-GPU Muon-LoRA recipe")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--muon-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--swift", default=shutil.which("swift") or "swift")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path, label in ((args.dataset, "dataset"), (args.muon_repo, "Muon repository")):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    command = build_command(args)
    print(subprocess.list2cmdline(command))
    if args.dry_run:
        return
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env.setdefault("NPROC_PER_NODE", "4")
    env.setdefault("MASTER_PORT", "29733")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env["MUON_REPO"] = str(args.muon_repo.resolve())
    env.setdefault("MUON_EAGER", "1")
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
