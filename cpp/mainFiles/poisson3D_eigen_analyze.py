#!/usr/bin/env python3
"""Solve the pencils exported by poisson3D_eigen_compare and compare ball modes.

Requires NumPy and SciPy. See README_poisson3D_eigen_compare.md.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import linalg, optimize, sparse, special
from scipy.sparse import linalg as spla


GROUPS = [(0, 1, 0, 1), (1, 1, 1, 4), (2, 1, 4, 9), (0, 2, 9, 10)]


def ball_spectrum(count: int, radius: float) -> np.ndarray:
    """Positive Dirichlet spectrum, including angular multiplicities 2*l+1."""
    spectrum = []
    for ell in range(count):
        roots_needed = max(1, (count + 2*ell)//(2*ell+1))
        left = 1e-4
        roots_found = 0
        while roots_found < roots_needed:
            right = left + np.pi/8
            if special.spherical_jn(ell, left)*special.spherical_jn(ell, right) < 0:
                root = optimize.brentq(lambda x: special.spherical_jn(ell, x), left, right,
                                       xtol=1e-13)
                spectrum.extend([(root/radius)**2]*(2*ell+1))
                roots_found += 1
            left = right
    return np.sort(spectrum)[:count]


def load_matrix(file: Path, n: int) -> sparse.csc_matrix:
    data = np.loadtxt(file, ndmin=2)
    if data.shape[1] != 3 or not np.all(np.isfinite(data)):
        raise ValueError(f"Invalid triplet matrix: {file}")
    ij = data[:, :2]
    if np.any(ij != np.floor(ij)) or np.min(ij) < 1 or np.max(ij) > n:
        raise ValueError(f"Invalid one-based indices: {file}")
    indices = ij.astype(np.int64)-1
    matrix = sparse.coo_matrix((data[:, 2], (indices[:, 0], indices[:, 1])), shape=(n, n)).tocsc()
    matrix.eliminate_zeros()
    return matrix


def symmetry_defect(matrix: sparse.spmatrix) -> float:
    return float(spla.norm(matrix-matrix.T)/max(spla.norm(matrix), np.finfo(float).tiny))


def positive_factor(matrix: sparse.csc_matrix, label: str):
    """Audit SPD using a symmetric permutation and unpivoted scalar LDL pivots.

    SuperLU represents the factor as LU. With identical row/column permutations
    and a symmetric input, the diagonal of U is D in LDL^T. Refuse a permutation
    mismatch rather than inferring inertia from a row-pivoted factorization.
    """
    factor = spla.splu(matrix, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.,
                       options={"SymmetricMode": True})
    if not np.array_equal(factor.perm_r, factor.perm_c):
        raise ValueError(f"{label}: cannot audit inertia with unequal row/column permutations")
    pivots = factor.U.diagonal()
    if not np.all(np.isfinite(pivots)) or np.any(pivots <= 0):
        raise ValueError(f"{label}: nonpositive LDL pivots; the positive ball comparison is invalid. "
                         "Check Nitsche coercivity, ghost stabilization, and cut degeneracy.")
    return factor


def subspace_metrics(vectors: np.ndarray, mass: sparse.spmatrix,
                     loads: np.ndarray, gram: np.ndarray) -> tuple[float, float, float]:
    """L2 principal angles; invariant under rotations/scalings within either space."""
    numerical_gram = vectors.T @ (mass @ vectors)
    ln = linalg.cholesky(numerical_gram, lower=True)
    le = linalg.cholesky(gram, lower=True)
    overlap = linalg.solve_triangular(le, loads.T @ vectors, lower=True)
    overlap = linalg.solve_triangular(ln, overlap.T, lower=True).T
    cosines = linalg.svdvals(overlap)
    if np.max(cosines) > 1.+1e-6:
        raise ValueError("Reference overlap exceeds one; check quadrature or mass conditioning")
    minimum = float(np.clip(np.min(cosines), 0., 1.))
    return minimum, float(np.sqrt(1.-minimum**2)), float(np.degrees(np.arccos(minimum)))


def write_csv(file: Path, rows: list[dict]) -> None:
    with file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(row: dict[str, str], directory: Path, prefix: str, nev: int,
            vtk_modes: int, tolerance: float) -> tuple[list[dict], list[dict]]:
    n = int(row["ndof"])
    if nev >= n:
        raise ValueError(f"--nev must be less than ndof={n}")
    matrices = {key: load_matrix(directory/row[key+"file"], n) for key in ("A", "B", "M", "T")}
    defects = {key: symmetry_defect(value) for key, value in matrices.items()}
    if max(defects.values()) > 1e-12:
        raise ValueError(f"Asymmetric exported matrices: {defects}")
    a, b, m, trace = (matrices[key] for key in ("A", "B", "M", "T"))
    factor = positive_factor(a, "Stiffness A")
    positive_factor(b, "Eigenproblem mass B")
    if row["Bfile"] != row["Mfile"]:
        positive_factor(m, "Physical mass M")
    # A is audited SPD above: shift-invert at zero returns the genuinely lowest
    # positive eigenvalues, rather than silently omitting negative Nitsche modes.
    inverse = spla.LinearOperator(a.shape, matvec=factor.solve, dtype=float)
    values, vectors = spla.eigsh(a, k=nev, M=b, sigma=0., which="LM", OPinv=inverse,
                                v0=np.random.default_rng(1729).normal(size=n), tol=tolerance)
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    norms = np.sqrt(np.sum(vectors*(m @ vectors), axis=0))
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Invalid physical L2 norms")
    vectors /= norms
    # Fix signs for reproducible visualization; eigenspace comparisons need no sign choice.
    for j in range(nev):
        if vectors[np.argmax(np.abs(vectors[:, j])), j] < 0:
            vectors[:, j] *= -1
    av, bv, mv = a @ vectors, b @ vectors, m @ vectors
    residuals = np.linalg.norm(av-bv*values, axis=0)/(np.linalg.norm(av, axis=0)+
                                                   np.abs(values)*np.linalg.norm(bv, axis=0))
    if not np.all(np.isfinite(values)) or np.max(residuals) > max(1e-8, 100*tolerance):
        raise ValueError(f"Eigenpair residual too large: {np.max(residuals):.3e}")
    r = float(row["radius"])
    exact = ball_spectrum(nev, r)
    boundary = np.sqrt(np.maximum(0., r*np.sum(vectors*(trace @ vectors), axis=0)))
    ghost_fraction = np.sum(vectors*(bv-mv), axis=0)/np.sum(vectors*bv, axis=0)
    common = {key: row[key] for key in ("method", "level", "nx", "h", "ndof", "radius",
                                       "shift", "penalty", "tau_stiffness", "tau_mass")}
    eigenrows = []
    for j, value in enumerate(values):
        eigenrows.append(dict(common, mode=j+1, eigenvalue=float(value), exact=float(exact[j]),
                              relative_error=float(abs(value/exact[j]-1.)),
                              relative_residual=float(residuals[j]),
                              scaled_boundary_L2=float(boundary[j]),
                              ghost_mass_fraction=float(ghost_fraction[j])))

    loads = np.loadtxt(directory/row["Ffile"], ndmin=2)
    gram = np.loadtxt(directory/row["Gfile"], ndmin=2)
    if loads.shape != (n, 10) or gram.shape != (10, 10):
        raise ValueError("Unexpected reference matrix dimensions")
    groups = []
    for ell, radial_index, start, stop in GROUPS:
        cosine, gap, angle = subspace_metrics(vectors[:, start:stop], m,
                                              loads[:, start:stop], gram[start:stop, start:stop])
        groups.append(dict(common, ell=ell, radial_index=radial_index, multiplicity=stop-start,
                           exact=float(exact[start]), computed_min=float(values[start]),
                           computed_max=float(values[stop-1]),
                           max_relative_error=float(np.max(np.abs(values[start:stop]/exact[start]-1.))),
                           min_cosine=cosine, subspace_gap=gap, max_angle_degrees=angle,
                           max_relative_residual=float(np.max(residuals[start:stop])),
                           max_scaled_boundary_L2=float(np.max(boundary[start:stop])),
                           relative_volume_error=1.-float(row["volume"])/(4.*np.pi*r**3/3.),
                           max_symmetry_defect=max(defects.values())))

    stem = directory/f"{prefix}_eigenvectors_ball_{row['method']}_{row['level']}"
    np.savez_compressed(f"{stem}.npz", eigenvalues=values, coefficients=vectors,
                        exact_eigenvalues=exact, residuals=residuals)
    np.savetxt(f"{stem}.dat", vectors[:, :vtk_modes], fmt="%.17g",
               header=f"{n} {vtk_modes} {row['h']} {r:.17g} {row['shift']}", comments="")
    return eigenrows, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=Path("."))
    parser.add_argument("--prefix", default="poisson_eigcmp", help="basename used by the C++ exporter")
    parser.add_argument("--nev", type=int, default=14, help="lowest eigenpairs, at least 10")
    parser.add_argument("--vtk-modes", type=int, default=10, help="number saved for C++ --vtk")
    parser.add_argument("--tol", type=float, default=1e-10)
    args = parser.parse_args()
    if args.nev < 10 or not 1 <= args.vtk_modes <= min(args.nev, 100) or not 0 < args.tol < 1:
        parser.error("Require nev >= 10, 1 <= vtk-modes <= min(nev,100), and 0 < tol < 1")
    directory = args.matrix_dir
    with (directory/f"{args.prefix}_manifest.csv").open(newline="") as handle:
        cases = list(csv.DictReader(handle))
    if not cases:
        raise ValueError("Empty manifest")
    eigenrows, groups = [], []
    print("method      nx   (l,n)    exact      computed range          max rel.err    L2 gap")
    for row in cases:
        eigen, cluster = analyze(row, directory, args.prefix, args.nev, args.vtk_modes, args.tol)
        eigenrows.extend(eigen)
        groups.extend(cluster)
        for g in cluster:
            print(f"{g['method']:10} {int(g['nx']):3}   ({g['ell']},{g['radial_index']}) "
                  f"{g['exact']:9.5f}  [{g['computed_min']:9.5f},{g['computed_max']:9.5f}]  "
                  f"{g['max_relative_error']:11.3e}  {g['subspace_gap']:9.3e}", flush=True)
    write_csv(directory/f"{args.prefix}_eigenvalues.csv", eigenrows)
    write_csv(directory/f"{args.prefix}_eigenspaces.csv", groups)
    print(f"Wrote {args.prefix}_eigenvalues.csv, {args.prefix}_eigenspaces.csv, and eigenvector files.")


if __name__ == "__main__":
    main()
