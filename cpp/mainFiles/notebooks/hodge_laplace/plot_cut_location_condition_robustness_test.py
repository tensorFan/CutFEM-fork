#!/usr/bin/env python3
"""Estimate 1-norm condition numbers and plot cut-position robustness.

The C++ test writes sparse matrices as one-based (row, column, value) triplets
and records their names and matrix sizes in kikuchi_cut_position_manifest.csv.

In /build run:
python3 ../cpp/mainFiles/notebooks/stokes/plot_cut_condition_test.py

or:
python3 ../cpp/mainFiles/notebooks/stokes/plot_cut_condition_test.py --manifest hodge_cut_position_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def read_exported_matrix(path: Path, matrix_size: int) -> sp.csc_matrix:
    raw = np.loadtxt(path, dtype=float)
    if raw.size == 0:
        raise ValueError(f"Matrix file is empty: {path}")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 3:
        raise ValueError(
            f"Expected three columns (row, column, value) in {path}, "
            f"got shape {raw.shape}."
        )

    rows = raw[:, 0].astype(np.int64) - 1
    cols = raw[:, 1].astype(np.int64) - 1
    vals = raw[:, 2]

    if rows.min() < 0 or cols.min() < 0:
        raise ValueError(f"Expected one-based indices in {path}.")
    if rows.max() >= matrix_size or cols.max() >= matrix_size:
        raise ValueError(
            f"An index in {path} exceeds the manifest matrix size {matrix_size}."
        )

    matrix = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(matrix_size, matrix_size),
    ).tocsc()
    matrix.sum_duplicates()
    return matrix


def estimate_condition_number_1(matrix: sp.csc_matrix) -> tuple[float, float, float]:
    """Return estimates of ||A||_1, ||A^{-1}||_1, and cond_1(A)."""
    norm_a = float(np.asarray(abs(matrix).sum(axis=0)).ravel().max())

    # SuperLU factors the indefinite saddle-point matrix. onenormest then
    # estimates the norm of the inverse through linear solves, without forming
    # a dense inverse.
    lu = spla.splu(matrix)
    inverse_operator = spla.LinearOperator(
        matrix.shape,
        matvec=lambda x: lu.solve(x),
        rmatvec=lambda x: lu.solve(x, trans="T"),
        dtype=matrix.dtype,
    )
    norm_inverse = float(spla.onenormest(inverse_operator))
    return norm_a, norm_inverse, norm_a * norm_inverse


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("kikuchi_cut_position_manifest.csv"),
    )
    parser.add_argument(
        "--x-axis",
        choices=("min_cut_fraction", "min_cut_volume"),
        default="min_cut_fraction",
    )
    parser.add_argument(
        "--output-prefix",
        default="kikuchi_cut_position_condition",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    base_dir = manifest_path.parent
    rows = load_manifest(manifest_path)

    results: list[dict[str, Any]] = []
    for row in rows:
        matrix_path = base_dir / row["matrix_file"]
        matrix_size = int(row["matrix_size"])
        matrix = read_exported_matrix(matrix_path, matrix_size)

        try:
            norm_a, norm_inverse, condition = estimate_condition_number_1(matrix)
            status = "ok"
        except RuntimeError as error:
            norm_a = math.nan
            norm_inverse = math.inf
            condition = math.inf
            status = f"factorization_failed: {error}"

        result = dict(row)
        result.update(
            norm_A_1=norm_a,
            norm_A_inverse_1_est=norm_inverse,
            condition_1_est=condition,
            status=status,
        )
        results.append(result)

        print(
            f"case={row['case']:>2s}  method={row['method']:<20s}  "
            f"min fraction={float(row['min_cut_fraction']):.6e}  "
            f"cond_1~{condition:.6e}"
        )

    output_csv = base_dir / f"{args.output_prefix}.csv"
    fieldnames = list(results[0].keys())
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    labels = {
        "no_stabilization": "no stabilization",
        "macro_stabilization": "macro stabilization",
    }
    markers = {
        "no_stabilization": "D",
        "macro_stabilization": "+",
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for method in ("no_stabilization", "macro_stabilization"):
        method_rows = [row for row in results if row["method"] == method]
        method_rows.sort(key=lambda row: float(row[args.x_axis]))

        x = np.array([float(row[args.x_axis]) for row in method_rows])
        y = np.array([float(row["condition_1_est"]) for row in method_rows])
        finite = np.isfinite(y) & (x > 0.0) & (y > 0.0)

        ax.loglog(
            x[finite],
            y[finite],
            marker=markers[method],
            linestyle="-" if method == "macro_stabilization" else "none",
            label=labels[method],
        )

    if args.x_axis == "min_cut_fraction":
        ax.set_xlabel(r"Smallest active cut fraction $\min_T |T\cap\Omega_h|/|T|$")
    else:
        ax.set_xlabel(r"Smallest active cut volume $\min_T |T\cap\Omega_h|$")
    ax.set_ylabel(r"Estimated $1$-norm condition number")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()

    output_png = base_dir / f"{args.output_prefix}.png"
    output_pdf = base_dir / f"{args.output_prefix}.pdf"
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)

    print(f"\nWrote {output_csv}")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
