# Maxwell eigenfunction on an unfitted spherical shell

`maxwell3D_kikuchi_eigenfunction.cpp` assembles, solves, and exports one Maxwell
eigenfunction in a single C++ executable. It uses an unfitted Ned0–P1
Kikuchi formulation with mixed ghost stabilization and symmetric Nitsche terms.
Python, PETSc, and SLEPc are not required for this example.

The default domain is

```text
Omega = {x : pi/5 < |x-c| < pi/3},    c = (pi/2, pi/2, pi/2).
```

Both boundaries are represented by the P1 interpolant of the positive signed
distance `min(r-pi/5, pi/3-r)`. This differs from the comparison driver's quartic
product level set: it reduces geometric distortion, so discrete eigenvalues
can also differ on the same background grid. The condition
`n x u = 0` is imposed with symmetric Nitsche terms, together with the scalar
pressure boundary terms in the existing comparison driver. A global multiplier
filters the radial harmonic field `(x-c)/|x-c|^3` using the stabilized mass matrix.
The background grid is slightly
offset to avoid sphere/grid-vertex coincidences; the physical shell is fixed.

## Build and run

With the existing configured build, from the repository root:

```sh
cmake -S . -B build -DUSE_UMFPACK=ON
cmake --build build --target maxwell3D_kikuchi_eigenfunction -j4
./build/bin/maxwell3D_kikuchi_eigenfunction --output build/shell_mode.vtk
```

The target is enabled when `CUTFEM_BUILD_MAIN=ON` and `USE_UMFPACK=ON`.
It runs on one MPI rank, including when the library was built with MPI.
For a fresh serial build without MUMPS:

```sh
cmake -S . -B build/serial -DUSE_MPI=OFF -DUSE_MUMPS=OFF \
    -DUSE_UMFPACK=ON -DCUTFEM_BUILD_MAIN=ON -DCUTFEM_CREATE_DOCS=OFF
cmake --build build/serial --target maxwell3D_kikuchi_eigenfunction -j4
./build/serial/bin/maxwell3D_kikuchi_eigenfunction --output build/serial/shell_mode.vtk
```

The default is a `19 x 19 x 19` background vertex grid, target eigenvalue `3.2`,
Nitsche penalty coefficient `10`, relative residual tolerance `1e-8`, and at most
500 inverse iterations. Use
`--nx` to refine, `--target` to seek another mode, and `--help` for all options:

```sh
./build/bin/maxwell3D_kikuchi_eigenfunction --nx 19 --target 3.2 \
    --output build/shell_mode_n19.vtk
```

The solver factors `A - target*B` once with UMFPACK, then performs inverse
iteration. It retains the singular pressure/multiplier mass blocks and checks
the full mixed residual `||Ax-lambda Bx||/(||Ax||+|lambda| ||Bx||)`.
A failure to converge returns a nonzero exit status without writing a new VTK.
Close eigenvalues can require more iterations or a different target. Only one
mode near the target is computed; within a repeated eigenspace its orientation
is not unique. With `eps=mu=1`, the reported `lambda` is frequency squared.

## View in ParaView

1. Open `build/shell_mode_surface.vtk` and click **Apply**.
2. Colour by **eigenfunction**, component **Magnitude**.
3. Add a **Clip** through the shell centre to expose the inner boundary.
4. Add a **Glyph** filter and select **eigenfunction** for the arrow orientation.

Each shell run writes a volume file and a companion `_surface.vtk` containing
only the actual inner and outer interface triangles. Use the companion for
**Surface** and **Clip** views. The volume file deliberately duplicates vertices
per tetrahedron; its Surface representation exposes internal cell faces and
can look corrugated. Use that file for volume **Slice** and **Glyph** filters.

Both files contain the vector `eigenfunction`, scalar `pressure`, and field-data
values `eigenvalue` and `relative_residual`. The eigenfunction has unit physical
L2 norm; pressure uses the same scaling. It is sampled on the tetrahedra of the
physical cut domain, with separate vertices per element to preserve normal
jumps in the Nedelec field. ParaView's vector magnitude can be used directly.
Coordinates and fields are written as double arrays with 17 significant digits
to preserve small cut cells.

For the optional cube comparison, use `--example cube --nx 7`; its default
target is `2`, and the top face cuts through the last background mesh layer.

## Expected field and accuracy

The first positive PEC shell eigenvalue is approximately `3.02580060864` for
these radii and `eps=mu=1`. Its eigenspace has three orientations. On either
sphere the field is normal to the boundary, with two broad magnitude lobes and
an equatorial zero band; `n x u = 0` does **not** require `u = 0`.
Ned0 fields have discontinuous normal components across tetrahedra, so some
elementwise variation remains in the exported field.

The optional checker compares the computed field with all three orientations
of this analytic eigenspace, using volume quadrature:

```sh
python3 cpp/mainFiles/check_maxwell_shell_mode.py build/shell_mode_n19.vtk
```

The checker requires NumPy and VTK Python bindings. It derives the reference
from the first `l=1` TM root of
`[x*j1(x)]'(k*ri)*[x*y1(x)]'(k*ro) - [x*y1(x)]'(k*ri)*[x*j1(x)]'(k*ro) = 0`.
Comparison takes place on the exported polyhedral domain, so it is a numerical
diagnostic rather than a rigorous error bound for the curved domain.

Runs with the corrected formulation, `--penalty 10`, and target `3.2`:

| nx | Computed eigenvalue | Relative eigenvalue error | L2 energy in analytic eigenspace |
|---:|---:|---:|---:|
| 13 | 4.34586 | 43.6% | 87.7% |
| 19 | 3.37005 | 11.4% | 94.7% |
| 25 | 3.37870 | 11.7% | 97.1% |
| 31 | 3.35671 | 10.9% | 98.1% |

The field improves under refinement, but the eigenvalue error is still
substantial and is not monotone across these grids. A small algebraic residual
does not imply an accurate continuous eigenvalue. The default is suitable for
inspecting the field; quantitative spectral work needs further convergence
checks. The coefficient is configurable with `--penalty`; using `100` made the
coarse shell overly stiff (about `5.12` at `nx=19` after the stabilization fixes).

## Boundary and stabilization corrections

The grid offset changes background vertex coordinates only. Both the shell
level set and the harmonic field remain centred at `(pi/2,pi/2,pi/2)`.
The interface assembler uses `-interface.normal()`, i.e. minus the level-set
gradient; this points outward from the positive region on both shell boundaries.

Each shell run now prints the tangential boundary L2 norm (using the actual
planar assembly normals), total boundary L2 norm, pressure L2 norm, stabilized
mass, and harmonic constraint residual. These supplement the algebraic residual;
they do not by themselves establish convergence to a physical Maxwell mode.

The offset was not the cause of the localized patches. Compared with the
initial version, this driver removes the extra positive
`jump(grad(p))*jump(grad(q))` patch term while retaining both mixed terms
`jump(grad(p))*jump(v)` and `jump(u)*jump(grad(q))`. That extra pressure term
relaxes the gradient constraint and produced nonphysical modes in this example.
This driver therefore differs from `assemble_kikuchi` in the comparison driver.

The stationary library patch loop now includes a cut/uncut pair regardless of
which element has the lower index. Previously it omitted 652 of 8398 patches at
`nx=19`. A P0 patch-mass regression checks the resulting matrix entries.
The harmonic row and column now use `B*h_h`, avoiding the previous mismatch
between face stabilization in the constraint and patch stabilization in `B`.

At `nx=19` the corrected run has tangential boundary L2 norm `0.0954`, total
boundary norm `2.015`, pressure L2 norm `0.0701`, stabilized mass `1.092`
(physical mass `1`), and harmonic constraint residual below `5e-15`.
