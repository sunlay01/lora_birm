#!/usr/bin/env python3

from __future__ import annotations

import csv
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = SCRIPT_DIR.parents[2]
FIG_DIR = ROOT / "docs" / "thesis" / "figures"

MAIN_METHOD = ROOT / "cifar" / "results" / "official_cifarmnist_main_method_summary.csv"
MAIN_SEED = ROOT / "cifar" / "results" / "official_cifarmnist_main_seed_details.csv"
SNAP_METHOD = ROOT / "cifar" / "results" / "official_cifarmnist_thesis312_snapshot_method_summary.csv"
SNAP_SEED = ROOT / "cifar" / "results" / "official_cifarmnist_thesis312_snapshot_seed_details.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def make_panel_a(ax: plt.Axes, main_method_rows: list[dict[str, str]], snap_method_rows: list[dict[str, str]]) -> None:
    points = {
        "ERM": {
            "runtime": float(next(r for r in main_method_rows if r["method"] == "ERM")["runtime_mean_sec"]),
            "best": float(next(r for r in main_method_rows if r["method"] == "ERM")["best_ood_acc_mean_pct"]),
            "color": "#9aa0a6",
        },
        "IRMv1": {
            "runtime": float(next(r for r in main_method_rows if r["method"] == "IRMv1")["runtime_mean_sec"]),
            "best": float(next(r for r in main_method_rows if r["method"] == "IRMv1")["best_ood_acc_mean_pct"]),
            "color": "#5f6368",
        },
        "BIRM": {
            "runtime": float(next(r for r in main_method_rows if r["method"] == "BIRM")["runtime_mean_sec"]),
            "best": float(next(r for r in main_method_rows if r["method"] == "BIRM")["best_ood_acc_mean_pct"]),
            "color": "#264653",
        },
        "IRMv1 -> LoRA-BIRM": {
            "runtime": float(snap_method_rows[0]["runtime_mean_sec"]),
            "best": float(snap_method_rows[0]["best_ood_acc_mean_pct"]),
            "color": "#c0392b",
        },
    }

    for name, item in points.items():
        size = 160 if "LoRA" in name else 120
        ax.scatter(item["runtime"], item["best"], s=size, color=item["color"], edgecolor="white", linewidth=1.2, zorder=3)
        dx = -18 if name == "BIRM" else 4
        dy = 0.65 if name == "ERM" else 0.45
        if name == "IRMv1 -> LoRA-BIRM":
            dy = -0.95
        ax.text(item["runtime"] + dx, item["best"] + dy, f"{name}\n{item['best']:.2f}%, {item['runtime']:.1f}s", fontsize=9, color=item["color"])

    birm = points["BIRM"]
    lora = points["IRMv1 -> LoRA-BIRM"]
    ax.annotate(
        "near-BIRM best region\nwith shorter runtime",
        xy=(lora["runtime"], lora["best"]),
        xytext=(170, 61.4),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#444444"},
        color="#444444",
    )
    ax.axhline(birm["best"], color="#264653", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_xlim(90, 370)
    ax.set_ylim(40, 63.8)
    ax.set_xlabel("Mean runtime per seed (s)")
    ax.set_ylabel("Mean best OOD accuracy (%)")
    ax.set_title("A. Runtime vs best OOD region", loc="left", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.18, linewidth=0.8)


def make_panel_b(ax: plt.Axes, snap_seed_rows: list[dict[str, str]], main_seed_rows: list[dict[str, str]]) -> None:
    seeds = [11, 17, 23, 29, 37]
    ordered = sorted(snap_seed_rows, key=lambda r: seeds.index(int(r["seed"])))
    y = list(range(len(ordered)))[::-1]

    best_vals = [float(r["best_ood_acc_pct"]) for r in ordered]
    final_vals = [float(r["final_ood_acc_pct"]) for r in ordered]
    labels = [f"seed {r['seed']}" for r in ordered]

    for yi, best, final in zip(y, best_vals, final_vals):
        ax.plot([final, best], [yi, yi], color="#b0b7c3", linewidth=2.0, zorder=1)

    ax.scatter(final_vals, y, s=90, color="#7f8c8d", edgecolor="white", linewidth=1.0, label="final", zorder=3)
    ax.scatter(best_vals, y, s=105, color="#c0392b", edgecolor="white", linewidth=1.0, label="best snapshot", zorder=4)

    birm_best_mean = float(next(r for r in read_csv(MAIN_METHOD) if r["method"] == "BIRM")["best_ood_acc_mean_pct"])
    birm_best_max = max(float(r["best_ood_acc_pct"]) for r in main_seed_rows if r["method"] == "BIRM")
    ax.axvline(birm_best_mean, color="#264653", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(birm_best_max, color="#264653", linestyle=":", linewidth=1.0, alpha=0.8)

    for yi, best, final in zip(y, best_vals, final_vals):
        ax.text(best + 0.25, yi + 0.06, f"{best:.1f}", fontsize=8.5, color="#7a1f17")
        ax.text(final - 0.15, yi - 0.32, f"{final:.1f}", fontsize=8.0, color="#606c76", ha="right")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(47, 67.8)
    ax.set_xlabel("OOD accuracy (%)")
    ax.set_title("B. Best snapshot vs final", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.18, linewidth=0.8)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7f8c8d", markeredgecolor="white", markeredgewidth=1.0, markersize=9, label="final"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#c0392b", markeredgecolor="white", markeredgewidth=1.0, markersize=9.5, label="best snapshot"),
        Line2D([0], [0], color="#264653", linestyle="--", linewidth=1.0, label="BIRM best mean"),
        Line2D([0], [0], color="#264653", linestyle=":", linewidth=1.0, label="BIRM best max"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8.5)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    main_method_rows = read_csv(MAIN_METHOD)
    main_seed_rows = read_csv(MAIN_SEED)
    snap_method_rows = read_csv(SNAP_METHOD)
    snap_seed_rows = read_csv(SNAP_SEED)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 10,
            "figure.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    make_panel_a(axes[0], main_method_rows, snap_method_rows)
    make_panel_b(axes[1], snap_seed_rows, main_seed_rows)

    pdf_path = FIG_DIR / "cifar_snapshot_story.pdf"
    png_path = FIG_DIR / "cifar_snapshot_story.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
