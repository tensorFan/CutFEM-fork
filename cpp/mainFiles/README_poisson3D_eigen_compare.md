# Dirichlet Poisson eigenvalues on an unfitted 3D ball

`poisson3D_eigen_compare.cpp` exports a scalar CutFEM eigenproblem in the same
matrix-export style as `maxwell3D_eigen_compare.cpp`.
`poisson3D_eigen_analyze.py` solves it with SciPy and compares both eigenvalues
and eigenspaces with analytic ball modes.

The runs below show substantial coarse-mesh errors, followed by convergence of
the first ten eigenvalues and their eigenspaces. They do **not** provide evidence
of a persistent wrong Poisson spectrum under Dirichlet conditions. This result
does not resolve the separate Maxwell experiments.

## Problem and discrete forms

The driver uses the positive convention

\[
 -\Delta u=\lambda u\quad\text{in }B_R(0),\qquad u=0\quad\text{on }\partial B_R(0).
\]

For the convention `Delta u = lambda u`, negate the reported eigenvalues;
the eigenspaces are unchanged.

The background is a Cartesian tetrahedral mesh of a box of side `2.5 R`.
`nx` is the number of nodes per axis, and `h=2.5 R/(nx-1)` is the grid step.
The box is translated by `shift*h*(1,1.3,1.7)`, keeping the ball fixed. The
default shift is `0.137`, avoiding systematic cuts through mesh vertices.

The interface is the zero set of the P1 interpolant of `R^2-|x|^2`. The retained
positive region, denoted by `Omega_h`, is an inscribed polyhedral approximation
to the ball. Thus refinement includes both FE and geometry errors. The whole
boundary is unfitted; no background-box boundary belongs to the domain.

For continuous P1 functions, the stiffness is symmetric Nitsche with a patch
ghost penalty:

\[
 A(u,v)=(\nabla u,\nabla v)_{\Omega_h}
 -(\partial_n u,v)_{\Gamma_h}-(u,\partial_n v)_{\Gamma_h}
 +\frac{\gamma}{h}(u,v)_{\Gamma_h}+\tau_A J_\nabla(u,v).
\]

The normal is outward from the retained positive region. In the current library,
interface assembly supplies `-interface.normal()`, which has precisely this
orientation for the chosen level set.

For each neighboring pair with at least one cut element, let `delta_P u` be
the difference of the two element polynomials extended to the union of the two
full background tetrahedra. Each pair is counted once. The patch forms are

\[
 J_\nabla(u,v)=\sum_P(\nabla\delta_P u,\nabla\delta_P v)_P,
 \qquad J_0(u,v)=\sum_P(\delta_P u,\delta_P v)_P.
\]

These use `addPatchStabilization`, as in the Maxwell comparison. They are
**volume patch integrals**, not value jumps across the common face: those face
value jumps vanish for continuous P1. The gradient and value forms have the
stiffness and mass scalings, respectively; no extra power of `h` is inserted.

The driver compares two pencils with the same stiffness:

| Method | Eigenproblem mass B |
| --- | --- |
| `physical` | `M(u,v)=(u,v)_Omega_h` |
| `stabilized` | `M(u,v)+tau_mass*J_0(u,v)` |

Defaults are `gamma=100`, `tau_A=1`, `tau_mass=1`, and quadrature degree 7.
There is no boundary penalty on the eigenproblem mass.

## Analytic spectrum and eigenspace diagnostic

Separation in spherical coordinates gives

\[
 u_{l,n,m}(r,\omega)=C_{l,n}j_l(a_{l,n}r/R)Y_{l,m}(\omega),
 \qquad \lambda_{l,n}=a_{l,n}^2/R^2,
\]

where `a_{l,n}` is the nth positive zero of the spherical Bessel function `j_l`.
The multiplicity is `2l+1`. The spherical Bessel differential equation,
definitions, and zeros are given in [NIST DLMF 10.47](https://dlmf.nist.gov/10.47)
and [10.58](https://dlmf.nist.gov/10.58).

The first ten modes on the unit ball are:

| (l,n) | Eigenvalue | Multiplicity | Spectral indices |
| --- | ---: | ---: | --- |
| (0,1) | 9.8696044011 | 1 | 1 |
| (1,1) | 20.1907285564 | 3 | 2–4 |
| (2,1) | 33.2174619143 | 5 | 5–9 |
| (0,2) | 39.4784176044 | 1 | 10 |

The C++ driver evaluates these analytic functions at physical quadrature points,
without interpolating them into P1. It exports
`F_ij=(phi_i,u_exact_j)_Omega_h` and
`G_ij=(u_exact_i,u_exact_j)_Omega_h`. The reference functions are normalized on
the exact ball, but `G` accounts for their restriction to the computed domain.

For each cluster, the Python script forms the overlap between orthonormal bases
of the exact and numerical eigenspaces in **physical L2(Omega_h)**. It uses `M`
for this comparison even when solving with stabilized mass `B`. If `s_min` is
the smallest singular value of this overlap, it reports

\[
 \texttt{subspace_gap}=\sqrt{1-s_{\min}^2}
 =\sin(\text{largest principal angle}).
\]

Zero means coincident spaces and one means a missing/orthogonal direction.
This comparison allows signs, arbitrary rotations, and different normalizations
within a multiple eigenspace. It does not penalize an eigenvector merely for
being a different linear combination of spherical harmonics.

Clusters use the spectral index ranges in the table. On very coarse meshes,
different clusters can overlap or reorder, and a gap near one can reflect that
unresolved ordering. The refinement sequence is essential to interpretation.

The analyzer also records `||Ax-lambda Bx||/(||Ax||+|lambda| ||Bx||)`,
`sqrt(R)*||u_h||_Gamma_h/||u_h||_Omega_h`, ghost-mass fractions, matrix symmetry,
and relative volume error. It audits positive definiteness of the stiffness and
mass matrices before using shift-invert at zero. An indefinite Nitsche matrix
is rejected instead of silently hiding negative modes and comparing only the
positive eigenvalues.

## Build and run

From the repository root, a serial build needs no external direct solver:

```sh
cmake -S . -B build/poisson-serial -DUSE_MPI=OFF -DUSE_MUMPS=OFF -DUSE_UMFPACK=OFF -DCUTFEM_CREATE_DOCS=OFF
cmake --build build/poisson-serial --target poisson3D_eigen_compare -j 4

python3 -m venv build/poisson-venv
build/poisson-venv/bin/python -m pip install numpy scipy

build/poisson-serial/bin/poisson3D_eigen_compare --prefix build/poisson-results/ball
build/poisson-venv/bin/python cpp/mainFiles/poisson3D_eigen_analyze.py --matrix-dir build/poisson-results --prefix ball
```

The default run uses `nx=9,17,33` and both mass choices. The analyzer computes
the lowest 14 eigenpairs and compares the first ten as four complete clusters.
Additional eigenpairs can be requested with `--nev`; eigenspace diagnostics
remain restricted to those first four clusters.

Output includes a manifest; one-based sparse triplets for `A`, `B`, physical
mass `M`, and boundary mass `T`; dense reference matrices `F,G`; eigenvalue and
eigenspace CSV files; and eigenvectors in `.npz` and text `.dat` files. Filenames
in the manifest are relative to its directory. Eigenvectors are normalized in
physical L2. A C++ build with MPI enabled is supported with **one rank**; the
exporter explicitly rejects multi-rank execution.

To visualize the computed eigenfunctions on the actual cut domain, rerun the
driver with the same geometry/mesh/mass options and `--vtk`:

```sh
build/poisson-serial/bin/poisson3D_eigen_compare --prefix build/poisson-results/ball --vtk
```

This reads the saved coefficients and writes VTK fields `mode_1` through
`mode_10`. It checks the DoF count, `h`, radius, and shift against the saved
geometry header. Open an output `.vtk` in ParaView and color by a mode.

Options include `--nx0`, `--levels`, `--radius`, `--penalty`, `--tau-stiffness`,
`--tau-mass`, `--mass physical|stabilized|both`, `--shift`, `--quadrature 5|7`,
and `--prefix`. Use `--help` for defaults. The output directory is created
automatically. Use a distinct prefix for each parameter study.

## Measured results

The following runs used the default parameters and the unit ball. The `nx=49`
run used `--nx0 49 --levels 1` with a separate prefix. The maximum eigenvalue
error covers spectral indices 1–10; the maximum gap covers all four clusters.

| Mass | nx | DoFs | First eigenvalue | Max relative eigenvalue error | Max L2 subspace gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| physical | 9 | 345 | 12.95091612 | 194.73% | 0.99996 |
| physical | 17 | 1819 | 10.35179218 | 34.26% | 0.75743 |
| physical | 33 | 11442 | 9.96330930 | 7.34% | 0.11743 |
| physical | 49 | 35521 | 9.90779234 | 2.86% | 0.04717 |
| stabilized | 9 | 345 | 6.34676211 | 40.76% | 0.99884 |
| stabilized | 17 | 1819 | 9.42346099 | 8.50% | 0.96806 |
| stabilized | 33 | 11442 | 9.86714361 | 3.83% | 0.06260 |
| stabilized | 49 | 35521 | 9.88140068 | 1.79% | 0.02588 |

The complete per-cluster results are stored in
[poisson_ball_eigen_compare_results.csv](../../Documentation/poisson_ball_eigen_compare_results.csv).
The assembly and solve artifacts from this session are under
`build/poisson-eigen/` (prefixes `default` and `fine`). The executable used for
these measurements is `build/obstacle-explore/bin/poisson3D_eigen_compare`.

At `nx=49`, physical-mass eigenvalue ranges are `20.29908–20.46462` for the
triplet and `33.58925–34.16628` for the quintuplet. Their L2 gaps are `0.00464`
and `0.04717`, respectively. Thus the repeated eigenvalues split on the finite
mesh, while their combined eigenspaces approach the analytic spaces.

The relative volume error decreases from `7.19%` to `1.82%` to `0.459%` to
`0.203%`. The largest scaled boundary norm among the first ten physical-mass
modes decreases from `0.1325` to `0.02824` to `0.00811` to `0.00370`. All reported
first-ten eigenpair residuals are below `6e-14`; all stiffness and mass audits
pass. Small solver residuals alone would not have detected the large coarse
PDE errors shown in the table.

Additional checks performed:

- Reducing **both** patch coefficients to `0.1` still gives convergence on
  `nx=9,17,33`; physical-mass first eigenvalues are `11.20679,10.17054,9.94319`.
- At `nx=33`, simultaneously changing `shift` to `-0.111` and the Nitsche
  coefficient to `50` gives first eigenvalues `9.96298` (physical) and `9.86681`
  (stabilized), close to the default results.
- Doubling the radius at `nx=17` divides the first 14 eigenvalues by four, to
  relative discrepancy below `1.1e-14` for both mass choices.
- Changing quadrature degree from 7 to 5 at `nx=17` changes the first 14
  eigenvalues by at most `1.4e-10` relative, and changes the reported gaps by
  less than `4e-6`.
- The deliberately underpenalized case `--nx0 9 --levels 1 --penalty 0.1`
  produces an indefinite stiffness matrix and is rejected by the audit.
- Both VTK paths were exercised. Four Python tests check the analytic spectrum
  and multiplicities, radius scaling, invariance under basis changes within an
  eigenspace, detection of a wrong eigenspace, and the coercivity audit:

  ```sh
  build/poisson-venv/bin/python cpp/test/test_poisson_eigen_analysis.py
  ```

These experiments concern the low spectrum of this P1 Nitsche/patch formulation
on these mesh sequences. They show converging discretization errors rather than
a persistent discrepancy, but do not establish a result for every CutFEM
formulation, cut position, or spectral range. In particular, the unusually small
first-eigenvalue error for stabilized mass at `nx=33` should not be interpreted
as equivalent accuracy of all its eigenfunctions.
