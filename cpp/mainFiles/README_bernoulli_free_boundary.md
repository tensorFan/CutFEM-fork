# Bernoulli free boundary CutFEM demo

This target implements a compact 2D test of the CutFEM/level-set steepest descent algorithm for the Bernoulli free boundary identification problem.

## Implemented variant

The executable `bernoulli_free_boundary` uses the circular inverse Bernoulli test:

- background domain: `\hat\Omega = [0,1]^2`,
- physical domain: `\Omega = \hat\Omega \setminus B_r((1/2,1/2))`, represented by a level set `phi < 0` in `\Omega`,
- exact free boundary: radius `r = 1/4`,
- default initial guess: radius `r = 1/8`,
- exact data: `u = 4*sqrt((x-0.5)^2 + (y-0.5)^2) - 1`, `f = -4/r`, and Cauchy data induced on the fixed square boundary.

The code assembles:

1. the primal CutFEM problem with Nitsche enforcement of `u=0` on the free boundary and ghost-penalty stabilization;
2. the adjoint problem for the fixed-boundary mismatch functional;
3. the continuous/domain shape derivative evaluated with the discrete primal/adjoint solution;
4. the `H^1` Riesz velocity on the background mesh;
5. a stabilized Crank--Nicolson finite-element update of the level set.

The implementation is standalone and deliberately avoids SuiteSparse/UMFPACK/MPI dependencies by using an internal Jacobi-preconditioned BiCGSTAB solver.

## Build

From the repository root:

```bash
cmake -S . -B build-bernoulli \
  -DCUTFEM_CREATE_DOCS=OFF \
  -DCUTFEM_BUILD_TESTS=OFF \
  -DCUTFEM_BUILD_EXAMPLE=OFF \
  -DCUTFEM_BUILD_MAIN=ON \
  -DUSE_MPI=OFF \
  -DUSE_UMFPACK=OFF \
  -DUSE_MUMPS=OFF
cmake --build build-bernoulli --target bernoulli_free_boundary -j 2
```

## Run the verification case

```bash
./build-bernoulli/bin/bernoulli_free_boundary \
  --nx 25 \
  --max-it 25 \
  --transport-steps 8 \
  --vtk-every 10 \
  --learning-rate 0.5 \
  --prefix build-bernoulli/bin/bernoulli_circle
```

Expected qualitative behavior: the cost decreases monotonically after a short transient and the estimated zero-level-set radius approaches `0.25`.

On the test run used for this patch, the history was approximately:

- initial `J = 1.240713e+01`, estimated radius `0.123754`,
- final `J = 1.136436e-03`, estimated radius `0.248218` after 25 iterations.

The run writes:

- `<prefix>_history.csv`, with columns `iter,J,logJ,estimated_radius,beta_H1_norm,transport_time`,
- `<prefix>_NNN.vtk`, with nodal fields `phi`, `u`, `p`, `beta_x`, and `beta_y`.

## Useful command-line options

```text
--nx <int>                 cells per coordinate direction, default 30
--max-it <int>             maximum optimization iterations, default 30
--tol <real>               stopping tolerance for J, default 1e-5
--r0 <real>                initial radius, default 0.125
--learning-rate <real>     level-set pseudo-time scale, default 0.5
--transport-steps <int>    Crank--Nicolson substeps, default 8
--vtk-every <int>          write VTK every N iterations, default 5
--prefix <path>            output prefix, default bernoulli_circle
```
