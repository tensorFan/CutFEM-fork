#!/usr/bin/env python3
"""Plot residual histories written by obstacle_map_initial_guess.

Usage: python3 tools/plot_obstacle_initial_guess.py RUN_DIRECTORY --output PREFIX
Requires Matplotlib. Writes PREFIX.png and PREFIX.pdf.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLES = {
    ("current", 0): ("Current", "#1e293b"),
    ("all_contact", 0): ("All contact", "#b8860b"),
    ("far_normal", 0): ("Far normal", "#2563eb"),
    ("oscillatory", 0): ("Oscillatory", "#dc2626"),
    ("angular_only", 0): ("Angular only", "#ea580c"),
    ("normal_only", 1): ("Normal only (seed 1)", "#0d9488"),
    ("random", 1): ("Random (seed 1)", "#7c3aed"),
    ("random", 2): ("Random (seed 2)", "#a855f7"),
    ("random", 3): ("Random (seed 3)", "#be185d"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output", type=Path, default=Path("initial_guess_convergence"))
    args = parser.parse_args()
    with (args.runs / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (args.runs / "iterations.csv").open() as stream:
        histories = list(csv.DictReader(stream))
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    panels = [("spherical_cap", 4), ("spherical_cap", 6),
              ("exterior_circle", 8), ("exterior_circle", 16)]
    handles = {}
    for axis, (benchmark, n) in zip(axes.flat, panels):
        cases = [r for r in rows if r["benchmark"] == benchmark and int(r["subdivisions"]) == n]
        if not cases:
            raise ValueError(f"Missing results for {benchmark}, n={n}")
        for row in cases:
            key = row["initial_guess"], int(row["seed"])
            label, color = STYLES[key]
            history = [v for v in histories if v["benchmark"] == benchmark
                       and int(v["subdivisions"]) == n
                       and v["initial_guess"] == key[0] and int(v["seed"]) == key[1]]
            iterations = [int(v["iteration"]) for v in history]
            residuals = [float(v["tangent_residual"]) + float(v["normal_residual"]) for v in history]
            if not history or history[-1]["event"] != "returned":
                raise ValueError(f"Missing returned state for {benchmark}, n={n}, {key}")
            line, = axis.semilogy(iterations, residuals, color=color, linewidth=1.6,
                                  label=label, alpha=.9)
            handles[key] = line
            if row["status"] != "CONVERGED":
                axis.plot(iterations[-1], residuals[-1], "x", color=color,
                          markersize=8, markeredgewidth=2)
        axis.axhline(float(cases[0]["tolerance"]), color="#64748b", linewidth=1, linestyle="--")
        title = "Spherical cap" if benchmark == "spherical_cap" else "Exterior circle"
        axis.set_title(f"{title}: {cases[0]['nodes']} nodes", fontsize=12)
        axis.set_xlabel("Completed outer iterations")
        axis.grid(True, which="major", alpha=.18)
        axis.set_ylim(1e-10, 1e8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Global KKT residual")
    figure.suptitle("Initial guess sensitivity with the original solver settings", fontsize=15, y=.99)
    figure.legend([handles[key] for key in STYLES], [STYLES[key][0] for key in STYLES],
                  loc="lower center", ncol=3, bbox_to_anchor=(.5, .025), frameon=False)
    figure.text(.5, .005, "Dashed line: solver tolerance.  Cross: LINE_SEARCH_FAILED; returned map remains feasible.",
                ha="center", fontsize=9, color="#475569")
    figure.tight_layout(rect=(0, .15, 1, .96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(args.output.with_suffix(f".{extension}"), dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
