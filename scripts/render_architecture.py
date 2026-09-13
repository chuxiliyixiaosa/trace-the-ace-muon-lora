"""Render the solution architecture figure used by the write-up."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures" / "system_architecture.png"


def box(ax, xy, width, height, text, color, *, edge="#273142", size=10):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color="#172033",
        linespacing=1.25,
    )


def arrow(ax, start, end, *, color="#556274", style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            linestyle=style,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.2, 5.0), dpi=220)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    box(ax, (0.25, 2.0), 1.55, 1.0, "Learning objective\n+ transcript", "#E8EEF5")
    box(ax, (2.1, 2.0), 1.65, 1.0, "Context policy\n10,240 tokens", "#FFF0C9")
    arrow(ax, (1.8, 2.5), (2.1, 2.5))

    ax.text(4.05, 4.42, "Competition pipeline", fontsize=11, fontweight="bold", color="#234F7D")
    box(ax, (4.05, 3.25), 1.85, 0.9, "AdamW\nQwen3-4B LoRA", "#DCECF8")
    box(ax, (6.35, 3.25), 1.65, 0.9, "Final-token\nrepresentation", "#E7F2FA")
    box(ax, (8.45, 3.25), 1.65, 0.9, "5-model\nXGBoost", "#DCEFD9")
    box(ax, (10.55, 3.25), 1.3, 0.9, "Grouped\nbeta", "#F1E7D7")
    arrow(ax, (3.75, 2.7), (4.05, 3.7))
    arrow(ax, (5.9, 3.7), (6.35, 3.7))
    arrow(ax, (8.0, 3.7), (8.45, 3.7))
    arrow(ax, (10.1, 3.7), (10.55, 3.7))

    ax.text(8.7, 2.58, "Completed matrix-aware system", fontsize=11, fontweight="bold", color="#146B55")
    box(ax, (4.05, 0.65), 1.85, 0.95, "AdamW Qwen\nLayer-28 features", "#DCECF8")
    box(ax, (6.35, 0.65), 1.65, 0.95, "5-model XGBoost\n+ beta (40%)", "#DCEFD9")
    box(ax, (4.05, 2.05), 1.85, 0.95, "Muon Qwen3-4B\n505 / 0 audit", "#CFEBDD")
    box(ax, (6.35, 2.05), 1.65, 0.95, "Classifier head\n+ beta (60%)", "#E1F3E9")
    box(ax, (8.7, 1.25), 1.55, 1.0, "Probability\nfusion", "#FFE4C7")
    box(ax, (10.75, 1.25), 2.05, 1.0, "0.590509 Log Loss\n0.657425 AUROC", "#FAD7D5", edge="#9C3D39", size=10.5)

    arrow(ax, (3.75, 2.3), (4.05, 2.53))
    arrow(ax, (3.75, 2.3), (4.05, 1.12))
    arrow(ax, (5.9, 2.53), (6.35, 2.53))
    arrow(ax, (5.9, 1.12), (6.35, 1.12))
    arrow(ax, (8.0, 2.53), (8.7, 1.85))
    arrow(ax, (8.0, 1.12), (8.7, 1.62))
    arrow(ax, (10.25, 1.75), (10.75, 1.75), color="#9C3D39")

    ax.text(
        0.25,
        0.23,
        "Competition path: final-token XGBoost.  Completed research path: optimizer, depth, and head diversity.",
        fontsize=9.2,
        color="#4B5668",
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
