#!/usr/bin/env python
"""Method schematic: region capture, the region-miss loss, and calibration.

Self-contained illustration drawn from synthetic shapes; it uses no
experimental data and is deterministic. Panel (a) shows how the prediction
mask grows with the threshold and when a ground-truth region counts as
captured; panel (b) shows how the calibration curve selects the threshold.

Writes ``results/figures/method_schematic.pdf``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from record.losses import capture_threshold
from record.paths import results_dir

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
PAGE_W = 7.16  # inches, full-width figure

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

GRID = 64


def _disc(cy: int, cx: int, radius: float) -> np.ndarray:
    """Binary disc on a fixed grid."""
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2


def _score_field() -> np.ndarray:
    """Smooth score field peaking on two blobs of unequal confidence."""
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    big = 0.95 * np.exp(-(((yy - 22) ** 2 + (xx - 20) ** 2) / (2 * 9.0 ** 2)))
    small = 0.55 * np.exp(-(((yy - 44) ** 2 + (xx - 46) ** 2) / (2 * 4.5 ** 2)))
    return np.maximum(big, small)


def _panel_masks(axes) -> None:
    """Panel (a): the mask grows with lambda; capture is decided per region."""
    truth = _disc(22, 20, 9.0) | _disc(44, 46, 4.5)
    regions = [_disc(22, 20, 9.0), _disc(44, 46, 4.5)]
    score = _score_field()
    rho = 0.5

    for ax, lam in zip(axes, (0.30, 0.55, 0.80)):
        mask = score >= 1.0 - lam
        ax.set_facecolor("white")
        ax.imshow(np.zeros((GRID, GRID)), cmap="Greys", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(~mask, mask.astype(float)),
                  cmap=matplotlib.colors.ListedColormap([PALETTE[0]]),
                  alpha=0.60, vmin=0, vmax=1)
        ax.contour(truth, levels=[0.5], colors="black", linewidths=1.2)
        labels = []
        for region in regions:
            covered = (mask & region).sum() / region.sum()
            labels.append(covered)
            ys, xs = np.nonzero(region)
            captured = covered >= capture_threshold(rho)
            ax.text(xs.mean(), ys.min() - 4,
                    f"{covered:.0%}", ha="center", va="bottom", fontsize=7,
                    color=PALETTE[2] if captured else PALETTE[3],
                    fontweight="bold")
        missed = sum(c < capture_threshold(rho) for c in labels)
        ax.set_title(rf"$\lambda={lam:.2f}$" "\n" rf"$L = {missed}/2$",
                     fontsize=7.5, linespacing=1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)


def _panel_calibration(ax) -> None:
    """Panel (b): the calibration curve and the selected threshold."""
    lam = np.linspace(0, 1, 401)
    # Illustrative decreasing risk curve; shape only, no data claim.
    risk = 0.62 * (1.0 - lam) ** 1.6
    alpha = 0.20
    lam_hat = float(lam[np.argmax(risk <= alpha)])

    ax.plot(lam, risk, color=PALETTE[0], lw=1.6,
            label=r"calibration risk $\widehat{R}(\lambda)$")
    ax.axhline(alpha, color=PALETTE[3], lw=1.0, ls="--")
    ax.axvline(lam_hat, color=PALETTE[2], lw=1.0, ls=":")
    ax.plot([lam_hat], [alpha], marker="o", ms=4, color=PALETTE[2])
    ax.text(0.02, alpha + 0.02, r"target $\alpha$", color=PALETTE[3], fontsize=7)
    ax.text(lam_hat + 0.02, 0.45, r"$\hat{\lambda}$", color=PALETTE[2], fontsize=8)
    ax.annotate("", xy=(lam_hat, 0.02), xytext=(1.0, 0.02),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.35"))
    ax.text((lam_hat + 1.0) / 2, 0.05, "valid", ha="center", fontsize=7,
            color="0.35")
    ax.set_xlabel(r"threshold $\lambda$  (larger marks more area)")
    ax.set_ylabel("fraction of regions missed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.65)
    ax.legend(loc="upper right", frameon=False)


def build() -> str:
    fig = plt.figure(figsize=(PAGE_W, 2.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2.1], wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    _panel_masks(axes)
    ax_cal = fig.add_subplot(gs[0, 3])
    _panel_calibration(ax_cal)
    ax_cal.text(-0.22, 1.06, "(b)", transform=ax_cal.transAxes,
                fontsize=8, fontweight="bold")

    handles = [
        Patch(facecolor="white", edgecolor="black", label="ground-truth region"),
        Patch(facecolor=PALETTE[0], alpha=0.60, label=r"marked area $M_\lambda$"),
    ]
    axes[1].legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=7,
                   ncol=2, handlelength=1.2, columnspacing=1.0)
    axes[0].text(-0.12, 1.22, "(a)", transform=axes[0].transAxes,
                 fontsize=8, fontweight="bold")

    out = results_dir("figures") / "method_schematic.pdf"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def main() -> None:
    path = build()
    manifest = {
        "figure": "method_schematic",
        "path": path,
        "inputs": "none (synthetic illustration)",
        "capture_level": 0.5,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(results_dir("figures") / "method_schematic.meta.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {path}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
