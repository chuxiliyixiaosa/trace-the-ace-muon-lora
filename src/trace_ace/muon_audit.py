"""ms-swift 4.4.2 plugin that verifies Muon owns every trainable matrix."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from swift.callbacks import TrainerCallback, callbacks_map
from swift.optimizers import optimizers_map
from swift.optimizers.muon import MuonOptimizerCallback


class MuonPathCompatibility(MuonOptimizerCallback):
    def create_optimizer(self, model=None):
        if not getattr(self.args, "local_repo_path", None):
            self.args.local_repo_path = os.environ["MUON_REPO"]
        optimizer = super().create_optimizer(model)
        if os.environ.get("MUON_EAGER", "1") == "1":
            import toy_train

            function = toy_train.zeropower_via_newtonschulz5
            toy_train.zeropower_via_newtonschulz5 = getattr(
                function, "_torchdynamo_orig_callable", function
            )
        return optimizer


optimizers_map["muon"] = MuonPathCompatibility


class MuonAuditCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        optimizer = kwargs["optimizer"]
        while hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        if type(optimizer).__name__ != "Muon":
            raise RuntimeError(f"expected Muon, got {type(optimizer).__name__}")

        names = {id(parameter): name for name, parameter in kwargs["model"].named_parameters()}
        groups = {"muon": [], "adamw": []}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                destination = "muon" if optimizer.state[parameter]["use_muon"] else "adamw"
                groups[destination].append(
                    {"name": names[id(parameter)], "shape": list(parameter.shape)}
                )
        if not groups["muon"] or groups["adamw"]:
            raise RuntimeError(
                f"expected Muon-only trainable matrices, got "
                f"muon={len(groups['muon'])}, adamw={len(groups['adamw'])}"
            )
        if state.is_world_process_zero:
            report = {
                "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
                "global_batch": (
                    args.world_size
                    * args.per_device_train_batch_size
                    * args.gradient_accumulation_steps
                ),
                "defaults": optimizer.defaults,
                "parameters": groups,
            }
            Path(args.output_dir, "muon_optimizer_audit.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"[MUON_VERIFIED] muon={len(groups['muon'])} adamw=0 "
                f"global_batch={report['global_batch']}",
                flush=True,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        for name in ("loss", "grad_norm", "learning_rate"):
            value = (logs or {}).get(name)
            if value is not None and not math.isfinite(float(value)):
                raise RuntimeError(f"nonfinite {name} at step {state.global_step}: {value}")

    def on_train_end(self, args, state, control, **kwargs):
        for name, parameter in kwargs["model"].named_parameters():
            if parameter.requires_grad and not torch.isfinite(parameter).all():
                raise RuntimeError(f"nonfinite trained parameter: {name}")


callbacks_map["muon_audit"] = MuonAuditCallback
