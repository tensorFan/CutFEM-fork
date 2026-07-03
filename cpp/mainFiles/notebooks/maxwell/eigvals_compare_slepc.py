#!/usr/bin/env python3
r"""
One-command SLEPc postprocessor for maxwell3D_eigen_compare.cpp.

1. Syntax:
python3 ../cpp/mainFiles/notebooks/eigvals_compare_slepc.py \
    --matrix-dir . \        // directory containing the .dat matrices and manifest
    --prefix eigcmp \       // eigcmp_manifest.csv, eigcmp_A_*.dat, eigcmp_B_*.dat etc
    --target 1.0 \          // pivot
    --nev 20 \              // #eigvals
    --print-nev 20          // print #eigvals
    --no-condense-3field    // optional flag to skip condensing the 3-field w-block

2. Typical workflow from the directory where the C++ driver wrote its .dat files:

conda activate fenicsx-env
python3 ../cpp/mainFiles/notebooks/eigvals_compare_slepc.py --matrix-dir . --prefix eigcmp --target 3.2 --nev 41 --no-condense-3field

(For MPI/SLEPc runs, use for example:
    mpiexec -n 1 python3 eigvals_compare_slepc.py --matrix-dir . --prefix eigcmp --target 3.2 --nev 41
) // Not necessary, no speedup

3. The script reads <prefix>_manifest.csv, loads every exported A/B pair, checks
relative symmetry defects, statically condenses the 3-field w-block by default,
chooses an appropriate SLEPc problem type, and writes

    <prefix>_eigenvalues.csv

The manifest contains both examples:

    cube             : the simply connected [0,pi]^3 reference problem;
    spherical_shell  : the concentric shell R_inner < |x-c| < R_outer.

For spherical_shell the C++ driver can add one Lagrange multiplier removing the
radial harmonic field h=(x-c)/|x-c|^3.  This makes B singular in the constrained rows,
so the script prefers GHIEP for symmetric singular/indefinite pencils when the
installed SLEPc exposes it; otherwise it falls back to GNHEP.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

comm = MPI.COMM_WORLD


# Canonical example names written by the current C++ driver.  The aliases are
# accepted only so that older manifests and command lines remain readable.
EXAMPLE_ALIASES = {
    "cube": "cube",
    "spherical_shell": "spherical_shell",
    "shell": "spherical_shell",
    "cube_hole": "spherical_shell",
    "hole": "spherical_shell",
}


def canonical_example_name(name: str) -> str:
    stripped = name.strip()
    return EXAMPLE_ALIASES.get(stripped, stripped)


def par_print(*args, **kwargs) -> None:
    if comm.rank == 0:
        print(*args, **kwargs)
        sys.stdout.flush()


@dataclass
class MatrixCase:
    example: str
    method: str
    level: int
    nx: int
    ny: int
    nz: int
    h: float
    Afile: str
    Bfile: str
    n0: int
    n1: int
    n2: int
    nlambda: int = 0
    harmonic_filter: bool = False
    example_note: str = ""
    bc_note: str = ""

    @classmethod
    def from_manifest_row(cls, row: dict[str, str]) -> "MatrixCase":
        return cls(
            example=canonical_example_name(row.get("example", "cube")),
            method=row["method"],
            level=int(row["level"]),
            nx=int(row["nx"]),
            ny=int(row["ny"]),
            nz=int(row["nz"]),
            h=float(row["h"]),
            Afile=row["Afile"],
            Bfile=row["Bfile"],
            n0=int(row["n0"]),
            n1=int(row["n1"]),
            n2=int(row["n2"]),
            nlambda=int(row.get("nlambda", "0") or 0),
            harmonic_filter=(row.get("harmonic_filter", "0") == "1"),
            example_note=row.get("example_note", ""),
            bc_note=row.get("bc_note", ""),
        )

    @property
    def expected_size(self) -> int:
        return self.n0 + self.n1 + self.n2 + self.nlambda


# First Maxwell PEC cavity eigenvalues on [0,pi]^3.  These are only a reference
# for the simple cube; the spherical shell has no closed-form reference here.
EXACT_CUBE_REFERENCE = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])


def reference_for_case(case: MatrixCase) -> np.ndarray:
    if case.example == "cube":
        return EXACT_CUBE_REFERENCE
    return np.array([], dtype=float)


def read_manifest(path: Path) -> list[MatrixCase]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    cases = [MatrixCase.from_manifest_row(row) for row in rows]
    cases.sort(key=lambda c: (c.example, c.level, c.method))
    return cases


def load_triplet_matrix(path: Path, shape: Optional[tuple[int, int]] = None) -> sp.csr_matrix:
    """Load matlab::Export triplets as a CSR matrix.

    The loader detects one-based indexing and accepts explicitly padded shapes,
    which is essential when B has zero rows for algebraic or Lagrange-multiplier
    variables.
    """
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")

    data = np.loadtxt(path)
    if data.size == 0:
        if shape is None:
            raise ValueError(f"Empty matrix file without explicit shape: {path}")
        return sp.csr_matrix(shape, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Expected at least three columns in {path}, got shape {data.shape}")

    rows = data[:, 0].astype(np.int64)
    cols = data[:, 1].astype(np.int64)
    vals = data[:, 2].astype(float)

    if rows.min() == 1 or cols.min() == 1:
        rows = rows - 1
        cols = cols - 1

    if shape is None:
        n = int(max(rows.max(), cols.max()) + 1)
        shape = (n, n)

    M = sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
    M.sum_duplicates()
    M.eliminate_zeros()
    return M

def ensure_explicit_diagonal(M: sp.spmatrix) -> sp.csr_matrix:
    """
    Ensure every diagonal position exists in the sparse pattern.

    This does not regularize the operator: missing diagonal entries are inserted
    with value exactly 0.0. This is needed because PETSc's native LU may fail
    during shift-invert if A - sigma B lacks structural diagonal entries in
    saddle-point pressure rows.
    """
    M = M.tolil(copy=True)
    n = min(M.shape)
    for i in range(n):
        if i not in M.rows[i]:
            M.rows[i].append(i)
            M.data[i].append(0.0)

    M = M.tocsr()
    M.sum_duplicates()
    M.sort_indices()

    # Important: do NOT call eliminate_zeros() here.
    return M

# def scipy_to_petsc(M: sp.spmatrix) -> PETSc.Mat:
#     M = M.tocsr()
#     indptr = M.indptr.astype(PETSc.IntType, copy=False)
#     indices = M.indices.astype(PETSc.IntType, copy=False)
#     data = M.data.astype(PETSc.ScalarType, copy=False)
#     P = PETSc.Mat().createAIJ(size=M.shape, csr=(indptr, indices, data), comm=comm)
#     P.assemble()
#     return P
def scipy_to_petsc(M: sp.spmatrix) -> PETSc.Mat:
    M = ensure_explicit_diagonal(M)

    indptr = M.indptr.astype(PETSc.IntType, copy=False)
    indices = M.indices.astype(PETSc.IntType, copy=False)
    data = M.data.astype(PETSc.ScalarType, copy=False)

    P = PETSc.Mat().createAIJ(
        size=M.shape,
        csr=(indptr, indices, data),
        comm=comm,
    )

    P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    P.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)

    P.assemble()
    return P

def sparse_fro_norm(M: sp.spmatrix) -> float:
    if M.nnz == 0:
        return 0.0
    return float(np.sqrt(np.sum(np.abs(M.data) ** 2)))


def relative_asymmetry(M: sp.spmatrix) -> float:
    denom = sparse_fro_norm(M)
    if denom == 0.0:
        return 0.0
    D = (M - M.T).tocsr()
    D.eliminate_zeros()
    return sparse_fro_norm(D) / denom


def maybe_symmetrize(M: sp.spmatrix, enabled: bool, tol: float, label: str) -> sp.csr_matrix:
    asym = relative_asymmetry(M)
    if enabled and asym <= tol:
        return (0.5 * (M + M.T)).tocsr()
    if enabled and asym > tol:
        par_print(f"  not symmetrizing {label}: asymmetry {asym:.3e} exceeds tolerance {tol:.3e}")
    return M.tocsr()


def condense_three_field(A: sp.csr_matrix, B: sp.csr_matrix, case: MatrixCase,
                         symmetrize_tol: float = 1e-10) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Static condensation of the 3-field w-block.

    Unknown order from the C++ driver is

        (w, u, p, lambda_h),

    with lambda_h present only for the spherical-shell harmonic constraint.  The
    first block row gives

        Aww*w + Awk*k = 0,   k=(u,p,lambda_h),

    hence w = -Aww^{-1}Awk k and the condensed matrix is

        Akk - Akw Aww^{-1} Awk.
    """
    nw, nu, np_, nl = case.n0, case.n1, case.n2, case.nlambda
    if nw <= 0 or nu <= 0 or np_ <= 0:
        raise ValueError(f"3-field case has invalid block sizes: {(nw, nu, np_, nl)}")
    n_total = nw + nu + np_ + nl
    if A.shape[0] < n_total or B.shape[0] < n_total:
        raise ValueError(
            f"Matrix shape too small for manifest block sizes: A={A.shape}, B={B.shape}, blocks={(nw, nu, np_, nl)}"
        )

    par_print(f"  condensing 3-field w-block: nw={nw}, keep={nu + np_ + nl} (u={nu}, p={np_}, lambda={nl})")

    w = slice(0, nw)
    k = slice(nw, n_total)

    Aww = A[w, w].tocsc()
    Awk = A[w, k].tocsc()
    Akw = A[k, w].tocsr()
    Akk = A[k, k].tocsr()

    lu = spla.splu(Aww)
    X = lu.solve(Awk.toarray())
    Acond = Akk - sp.csr_matrix(Akw @ X)
    Bcond = B[k, k].tocsr()

    Acond = maybe_symmetrize(Acond, enabled=True, tol=symmetrize_tol, label="condensed A")
    Bcond = maybe_symmetrize(Bcond, enabled=True, tol=symmetrize_tol, label="condensed B")
    return Acond, Bcond


def matrix_has_structural_zero_rows(M: sp.csr_matrix) -> bool:
    row_nnz = np.diff(M.indptr)
    return bool(np.any(row_nnz == 0))


def choose_problem_type(hermitian: bool, constrained_or_singular: bool) -> tuple[object, str]:
    if not hermitian:
        return SLEPc.EPS.ProblemType.GNHEP, "GNHEP"
    if constrained_or_singular:
        # GHIEP is the right SLEPc category for a Hermitian generalized problem
        # with indefinite/singular B.  Some installations may not expose it in
        # slepc4py; GNHEP is a safe fallback.
        try:
            return SLEPc.EPS.ProblemType.GHIEP, "GHIEP"
        except AttributeError:
            return SLEPc.EPS.ProblemType.GNHEP, "GNHEP"
    return SLEPc.EPS.ProblemType.GHEP, "GHEP"


def solve_eigenproblem(A: sp.csr_matrix, B: sp.csr_matrix, *, target: float, nev: int,
                       tol: float, max_it: int, hermitian_tol: float,
                       constrained_or_singular: bool,
                       force_nonhermitian: bool = False,
                       diagnostic_blocks: Optional[list[tuple[str, int, int]]] = None
                       ) -> tuple[
                           list[complex],
                           dict[str, float | str | int],
                           Optional[dict[str, object]],
                       ]:
    asym_A = relative_asymmetry(A)
    asym_B = relative_asymmetry(B)
    hermitian = (asym_A <= hermitian_tol and asym_B <= hermitian_tol and not force_nonhermitian)

    # Detect an obviously singular B even when the manifest did not announce it.
    constrained_or_singular = constrained_or_singular or matrix_has_structural_zero_rows(B)
    problem_type, problem_type_name = choose_problem_type(hermitian, constrained_or_singular)

    Ap = scipy_to_petsc(A)
    Bp = scipy_to_petsc(B)

    eps = SLEPc.EPS().create(comm)
    eps.setOperators(Ap, Bp)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setProblemType(problem_type)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(target)
    eps.setDimensions(nev, PETSc.DECIDE, PETSc.DECIDE)
    eps.setTolerances(tol=tol, max_it=max_it)

    # st = eps.getST()
    # st.setType(SLEPc.ST.Type.SINVERT)
    # st.setShift(target)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target)

    ksp = st.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)

    # Prefer robust sparse LU with pivoting for saddle-point pencils.
    # for solver in ("mumps", "superlu", "superlu_dist"):
    #     try:
    #         pc.setFactorSolverType(solver)
    #         print(f"  using LU factor solver: {solver}")
    #         break
    #     except PETSc.Error:
    #         pass
    try:
        pc.setFactorSolverType("mumps")
        par_print("  using LU factor solver: mumps")
    except PETSc.Error:
        par_print("  MUMPS unavailable, using PETSc default LU")

    opts = PETSc.Options()
    opts["st_mat_mumps_icntl_14"] = "500"  # increase estimated workspace by 500%
    opts["st_mat_mumps_icntl_24"] = "1"    # null-pivot detection, useful diagnostics

    # Allow PETSc/SLEPc command-line options to override this script.
    eps.setFromOptions()
    eps.solve()

    nconv = eps.getConverged()
    vals = [eps.getEigenvalue(i) for i in range(nconv)]

    # Minimal diagnostic for the three-field formulation: inspect the returned
    # eigenvector whose eigenvalue has the smallest magnitude.  The block
    # fractions show whether the vector is carried by u (a physical harmonic
    # mode) or almost entirely by p/alpha (an algebraic mode in ker(B)).
    zero_diagnostic: Optional[dict[str, object]] = None
    if diagnostic_blocks and vals:
        i0 = min(range(len(vals)), key=lambda i: abs(vals[i]))
        z0 = vals[i0]

        xr = Ap.createVecRight()
        xi = Ap.createVecRight()
        xr.set(0.0)
        xi.set(0.0)
        eps.getEigenpair(i0, xr, xi)

        xnorm = math.hypot(xr.norm(), xi.norm())

        Bxr = Bp.createVecLeft()
        Bxi = Bp.createVecLeft()
        Bp.mult(xr, Bxr)
        Bp.mult(xi, Bxi)

        b_quadratic = float(np.real(xr.dot(Bxr) + xi.dot(Bxi)))
        relative_b_mass = math.sqrt(abs(b_quadratic)) / max(xnorm, 1.0e-300)

        block_fractions: dict[str, float] = {}
        if comm.size == 1:
            ar = xr.getArray(readonly=True)
            ai = xi.getArray(readonly=True)
            for name, begin, size in diagnostic_blocks:
                if size <= 0:
                    block_fractions[name] = 0.0
                    continue
                end = begin + size
                block_sq = float(
                    np.vdot(ar[begin:end], ar[begin:end]).real
                    + np.vdot(ai[begin:end], ai[begin:end]).real
                )
                block_fractions[name] = math.sqrt(max(block_sq, 0.0)) / max(
                    xnorm, 1.0e-300
                )

        zero_diagnostic = {
            "eigenvalue": z0,
            "relative_b_mass": relative_b_mass,
            "block_fractions": block_fractions,
        }

        xr.destroy()
        xi.destroy()
        Bxr.destroy()
        Bxi.destroy()

    info = {
        "problem_type": problem_type_name,
        "asym_A": asym_A,
        "asym_B": asym_B,
        "nconv": nconv,
        "iterations": eps.getIterationNumber(),
        "constrained_or_singular": int(constrained_or_singular),
    }

    eps.destroy()
    Ap.destroy()
    Bp.destroy()
    return vals, info, zero_diagnostic


def nearest_reference_error(x: complex, reference: np.ndarray) -> float:
    if len(reference) == 0:
        return math.nan
    return float(np.min(np.abs(reference - x.real)))


def process_case(case: MatrixCase, matrix_dir: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    A_path = matrix_dir / case.Afile
    B_path = matrix_dir / case.Bfile

    A = load_triplet_matrix(A_path)
    expected = case.expected_size
    if expected > 0 and A.shape[0] < expected:
        # This should rarely be needed, but it protects against exporters that
        # skip explicit zero diagonals.
        A = load_triplet_matrix(A_path, shape=(expected, expected))
    B = load_triplet_matrix(B_path, shape=A.shape)

    par_print("\n" + "=" * 78)
    par_print(
        f"{case.example}/{case.method}, level {case.level}, mesh {case.nx}x{case.ny}x{case.nz}, h={case.h:.6g}"
    )
    if case.example_note:
        par_print(f"  example: {case.example_note}")
    if case.bc_note:
        par_print(f"  bc: {case.bc_note}")
    if case.harmonic_filter:
        par_print("  harmonic filter: enforcing (u,h)_Omega=0")
    par_print(f"  block sizes: n0={case.n0}, n1={case.n1}, n2={case.n2}, nlambda={case.nlambda}")
    par_print(f"  raw size: A={A.shape}, B={B.shape}")
    par_print(f"  raw asymmetry: A={relative_asymmetry(A):.3e}, B={relative_asymmetry(B):.3e}")

    # constrained_or_singular = case.nlambda > 0 or case.method == "3field"
    constrained_or_singular = (
        case.nlambda > 0
        or case.method in {"kikuchi", "3field"}
        or matrix_has_structural_zero_rows(B)
    )
    force_nonhermitian = args.force_nonhermitian

    three_field_condensed = False
    if case.method == "3field" and not args.no_condense_3field:
        A, B = condense_three_field(A, B, case, symmetrize_tol=args.hermitian_tol)
        three_field_condensed = True
        par_print(f"  condensed size: A={A.shape}, B={B.shape}")
        par_print(f"  condensed asymmetry: A={relative_asymmetry(A):.3e}, B={relative_asymmetry(B):.3e}")

    diagnostic_blocks: Optional[list[tuple[str, int, int]]] = None
    if case.method == "3field":
        if three_field_condensed:
            # Condensed ordering: (u, p, alpha).
            diagnostic_blocks = [
                ("u", 0, case.n1),
                ("p", case.n1, case.n2),
                ("alpha", case.n1 + case.n2, case.nlambda),
            ]
        else:
            # Raw ordering: (w, u, p, alpha).
            diagnostic_blocks = [
                ("w", 0, case.n0),
                ("u", case.n0, case.n1),
                ("p", case.n0 + case.n1, case.n2),
                ("alpha", case.n0 + case.n1 + case.n2, case.nlambda),
            ]

    vals, info, zero_diagnostic = solve_eigenproblem(
        A, B,
        target=args.target,
        nev=args.nev,
        tol=args.tol,
        max_it=args.max_it,
        hermitian_tol=args.hermitian_tol,
        constrained_or_singular=constrained_or_singular,
        force_nonhermitian=force_nonhermitian,
        diagnostic_blocks=diagnostic_blocks,
    )

    vals_sorted = sorted(vals, key=lambda z: (z.real, abs(z.imag)))
    reference = reference_for_case(case)

    par_print(
        f"  solver: {info['problem_type']}, nconv={info['nconv']}, its={info['iterations']}, "
        f"asym_A={info['asym_A']:.3e}, asym_B={info['asym_B']:.3e}"
    )
    if len(reference):
        par_print(f"  cube exact reference: {np.array2string(reference, precision=2)}")
    else:
        par_print("  no closed-form reference eigenvalues supplied for this example")
    par_print("  computed real parts:")
    par_print("   ", np.array2string(np.array([z.real for z in vals_sorted[:args.print_nev]]), precision=6))

    if zero_diagnostic is not None:
        z0 = complex(zero_diagnostic["eigenvalue"])
        bmass = float(zero_diagnostic["relative_b_mass"])
        fractions = zero_diagnostic["block_fractions"]

        par_print("  smallest-|lambda| three-field mode:")
        par_print(f"    lambda = {z0.real:.6e}{z0.imag:+.2e}i")
        par_print(f"    relative u mass sqrt(|x^* B x|)/||x|| = {bmass:.3e}")

        if fractions:
            formatted = ", ".join(
                f"{name}={float(value):.3e}"
                for name, value in fractions.items()
            )
            par_print(f"    coefficient fractions: {formatted}")

            u_fraction = float(fractions.get("u", 0.0))
            if bmass < 1.0e-8 and u_fraction < 1.0e-6:
                par_print("    diagnosis: probable algebraic p/alpha mode (u is essentially zero)")
            else:
                par_print("    diagnosis: mode carries nontrivial u mass")
        else:
            par_print("    block fractions require a serial run (-n 1)")

    rows: list[dict[str, object]] = []
    for j, z in enumerate(vals_sorted):
        rows.append({
            "example": case.example,
            "method": case.method,
            "level": case.level,
            "j": j,
            "eigenvalue_real": z.real,
            "eigenvalue_imag": z.imag,
            "nearest_exact_error": nearest_reference_error(z, reference),
            "problem_type": info["problem_type"],
            "asym_A": info["asym_A"],
            "asym_B": info["asym_B"],
            "nconv": info["nconv"],
            "iterations": info["iterations"],
            "matrix_size": A.shape[0],
            "nlambda": case.nlambda,
            "harmonic_filter": int(case.harmonic_filter),
        })
    return rows


def write_results_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if comm.rank != 0:
        return
    if not rows:
        par_print("No eigenvalue rows to write.")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    par_print(f"\nWrote eigenvalue summary: {path}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute all exported Maxwell eigenvalue comparisons with SLEPc.")
    parser.add_argument("--matrix-dir", default=".", help="directory containing the .dat matrices and manifest")
    parser.add_argument("--prefix", default="eigcmp", help="file prefix used by the C++ exporter")
    parser.add_argument("--manifest", default=None, help="explicit manifest path; default is <matrix-dir>/<prefix>_manifest.csv")
    parser.add_argument("--target", type=float, default=3.2, help="shift-invert target")
    parser.add_argument("--nev", type=int, default=41, help="number of eigenvalues requested")
    parser.add_argument("--print-nev", type=int, default=30, help="number of eigenvalues printed per case")
    parser.add_argument("--tol", type=float, default=1e-12, help="SLEPc convergence tolerance")
    parser.add_argument("--max-it", type=int, default=500, help="SLEPc maximum iterations")
    parser.add_argument("--hermitian-tol", type=float, default=1e-10, help="relative asymmetry tolerance for Hermitian problem types")
    parser.add_argument("--force-nonhermitian", action="store_true", help="always use GNHEP, even when symmetry checks pass")
    parser.add_argument("--no-condense-3field", action="store_true", help="solve the raw 3-field pencil instead of condensing w")
    parser.add_argument("--methods", default="all", help="comma-separated methods to run, or all")
    parser.add_argument(
        "--examples",
        default="all",
        help="comma-separated examples to run (cube,spherical_shell), or all",
    )
    parser.add_argument("--levels", default="all", help="comma-separated levels to run, or all")
    parser.add_argument("--output", default=None, help="CSV output path; default <matrix-dir>/<prefix>_eigenvalues.csv")
    return parser.parse_args(argv)


def filter_cases(cases: list[MatrixCase], args: argparse.Namespace) -> list[MatrixCase]:
    if args.methods != "all":
        methods = {m.strip() for m in args.methods.split(",") if m.strip()}
        cases = [c for c in cases if c.method in methods]
    if args.examples != "all":
        examples = {
            canonical_example_name(e)
            for e in args.examples.split(",")
            if e.strip()
        }
        cases = [c for c in cases if c.example in examples]
    if args.levels != "all":
        levels = {int(x.strip()) for x in args.levels.split(",") if x.strip()}
        cases = [c for c in cases if c.level in levels]
    return cases


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    matrix_dir = Path(args.matrix_dir)
    manifest_path = Path(args.manifest) if args.manifest else matrix_dir / f"{args.prefix}_manifest.csv"
    output_path = Path(args.output) if args.output else matrix_dir / f"{args.prefix}_eigenvalues.csv"

    cases = read_manifest(manifest_path)
    cases = filter_cases(cases, args)
    if not cases:
        par_print("No manifest rows matched the requested examples/methods/levels.")
        return 1

    par_print(f"Read {len(cases)} matrix cases from {manifest_path}")
    par_print(f"SLEPc target={args.target}, nev={args.nev}, tol={args.tol}")

    all_rows: list[dict[str, object]] = []
    for case in cases:
        all_rows.extend(process_case(case, matrix_dir, args))

    write_results_csv(output_path, all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
