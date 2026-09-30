# Obstacle maps with fitted coordinate finite elements

`cpp/mainFiles/obstacle_map.cpp` is the example declaration. It selects a mesh,
target geometry, prescribed trace, feasible initial map, solver options, and
reference solution through `ObstacleMapExample<D>`. The numerical implementation
is separate:

| File | Responsibility |
| --- | --- |
| `cpp/problem/obstacleMap.hpp` | Library P1 space, chart-dependent weak-form assembly, feasibility, collar and bulk updates, and global KKT residuals |
| `cpp/problem/obstacleTarget.hpp` | Full target metric and its derivatives from an embedding, Jacobian, and Hessian |
| `cpp/solver/boundQuadratic.hpp` | Reusable convex quadratic solver with lower bounds, active constraints, and reduced preconditioned CG |
| `cpp/example/obstacle_map/benchmarks.hpp` | Library box meshes, ball mapping, target charts, independent initial fields, and reference profiles |
| `cpp/example/obstacle_map/driver.hpp` | Fluent example interface, arguments, reference error, and output |

The implementation is serial. It accepts fitted triangular sources (`D=2`) and
tetrahedral sources (`D=3`), with target coordinate dimension `m` chosen at run
time. Its geometry interface supports a full target metric in one certified
coordinate representation. This may be a regular covering parameterization
when the entire map has a single continuous coordinate lift. It does not
implement a target atlas or promise to cover every target topology with one
coordinate field.

## Energy and weak formulation

Write target coordinates as

\[
q=(s^1,\ldots,s^{m-1},r),\qquad r\geq0,\qquad u=F(q).
\]

The coordinate domain flattens the obstacle to `r=0`. In a declared collar,
`r` is inward geodesic distance to the target boundary. The same representation
must extend over the rest of the computed image, where `r` may be a nonnegative
defining coordinate. This extension is an explicit requirement on the geometry
provider; extending a Fermi map through a cut locus is not valid. The circle and
spherical examples permit unwrapped longitude on the real line. Their physical
parameterizations are periodic coverings, so they do not claim global
injectivity. The globally defined scalar longitude is a chosen continuous lift.

Let `g` be the source metric and `G(q)` the full `m` by `m` target metric. We use
the half Dirichlet energy

\[
 E(q)=\frac12\int_\Omega
 G_{ab}(q)\,g^{ij}\partial_iq^a\partial_jq^b\,dV_g.
\]

Indices repeated in a formula are summed. For variations vanishing on the
prescribed source boundary, its exact first variation is

\[
 DE(q)[v]=\int_\Omega
 G_{cb}(q)\langle\nabla v^c,\nabla q^b\rangle_g
 +\frac12\partial_cG_{ab}(q)\,v^c
          \langle\nabla q^a,\nabla q^b\rangle_g\,dV_g.
\]

Every first derivative of the metric participates, including derivatives with
respect to tangential coordinates and derivatives of metric cross terms.
`assemble()` implements this variation directly. Its frozen-metric stiffness is
a positive step metric, not the full Hessian of the nonlinear energy.

Inside a Fermi collar, `G=diag(A,1)`, where `A` is an `(m-1)` by `(m-1)` matrix.
Writing the normal test as `eta` and a tangential test as `xi`, the forms reduce to

\[
 g_r(\eta)=\int_\Omega
 \langle\nabla r,\nabla\eta\rangle_g
 +\frac12\partial_r A_{ab}
      \langle\nabla s^a,\nabla s^b\rangle_g\eta\,dV_g,
\]

\[
 g_{s^c}(\xi)=\int_\Omega
 A_{cb}\langle\nabla s^b,\nabla\xi\rangle_g
 +\frac12\partial_c A_{ab}
      \langle\nabla s^a,\nabla s^b\rangle_g\xi\,dV_g.
\]

The scalar coefficient `A(s,r)` in the earlier split algorithm is the case
`m=2`. A single scalar coefficient cannot describe general higher dimensional
target boundaries.

Tangential stationarity is `g_s=0`. The normal variational inequality is
`g_r(z-r)>=0` for admissible nonnegative `z` with the same trace. In the collar,
the reaction has the sign convention

\[
 \lambda=-\Delta_g r+
 \frac12\partial_rA_{ab}\langle\nabla s^a,\nabla s^b\rangle_g
 \geq0,\qquad r\lambda=0.
\]

The manuscript convention `b=-Hess(r)` gives
`lambda=-(Delta_g r+H_r(ds,ds))`. On the noncontact set the complete first
variation vanishes: this is the harmonic map equation in the full metric.
It becomes componentwise Laplace equations in Euclidean Cartesian target
coordinates; that simplification does not hold in general coordinates.

## Discrete map and geometry contract

Coordinates belong to the conforming P1 source space:

\[
 q_h=\sum_i q_i\phi_i,\qquad u_h=F(q_h).
\]

The physical map is evaluated after coordinate interpolation. In particular,
`u_h` is generally not a piecewise affine ambient vector field. Barycentric P1
basis functions are nonnegative on their simplices, so nonnegative nodal `r_i`
imply nonnegative `r_h` everywhere. This implication would fail for many higher
order nodal bases. Interpolating feasible ambient nodal values directly can also
cross an exterior obstacle between nodes.

`TargetChart` requires:

| Member | Contract |
| --- | --- |
| `dimension` | Number `m` of target coordinates; the final coordinate is `r` |
| `tube_radius` | Positive threshold for the geometric collar |
| `metric(q,G,dG)` | `G[a*m+b]` and `dG[c][a*m+b]=partial_c G_ab`, all in the same coordinate system |
| `embed(q)` | Physical target map `F(q)` for output and comparison |
| `admissible_simplex(vertices)` | Certificate that the entire convex hull lies in the valid regular coordinate domain and maps into the feasible target |

The certificate must cover the whole element, not just its vertices or the
integration points. Vertex tests suffice when they establish containment in a
known convex coordinate region. The reparameterization helper uses a containing
prism because the nonlinear image of a convex hull need not equal the convex
hull of the transformed vertices.

The solver checks finite coordinates, nodal positivity, the supplied certificate,
and symmetric positive definite metrics at integration points. It cannot prove
that a user callback is a valid global certificate, that its derivatives are
correct, or that its metric and embedding describe the same geometry. Those are
provider responsibilities. For a covering parameterization, supplying a
compatible global coordinate lift is also a responsibility of problem setup;
the element certificate does not infer the topology of a prescribed physical
boundary map. A Euclidean obstacle level set alone does not provide
intrinsic distance or the full geometry of a curved ambient target.

For a target embedded in Euclidean space, `embeddedTargetChart()` provides a
reusable way to build this interface. Supply `EmbeddingJet {value, jacobian,
hessian}` for `F(q)`, with layouts `J[p*m+a]` and `H[(p*m+a)*m+b]`, plus the
collar radius and the whole-element certificate. The helper computes

\[
 G_{ab}=\sum_p J_{pa}J_{pb},\qquad
 \partial_cG_{ab}=\sum_p(H_{pac}J_{pb}+J_{pa}H_{pbc}).
\]

Both metric and output embedding then come from the same geometry callback.
This supports variable curvature and general matrix metrics without writing
problem-specific energy assembly. Tests use a noncircular parabola collar and a
three-dimensional target embedded in four dimensions. The helper checks finite
data, jet sizes, Hessian symmetry, and metric regularity; it does not construct
Fermi coordinates or infer reach. For a non-Euclidean ambient manifold, provide
the intrinsic metric and its derivatives directly through `TargetChart`.
Rejection of an uncertified whole element does not construct an alternative
chart. The implementation supports many geometries within a supplied regular
representation; it is not a globally universal geometry solver.

`ObstacleMapProblem<D>` takes a `std::shared_ptr<const Mesh2>` or
`std::shared_ptr<const Mesh3>` and retains it for the lifetime of its
`GFESpace<Mesh>`. The mesh must not be modified while the problem is in use.
It uses `DataFE<Mesh>::P1`, `GFElement::BF`, library local-to-global DOF numbering,
element measures and coordinate maps. Boundary vertices come from the native
mesh boundary elements. `space()` and `mesh()` expose these library objects;
`isBoundary(i)` reports the prescribed vertices. The old `SimplexMesh` and
`fittedSimplexMesh` copy adapter are no longer needed.

Assembly writes the library's `Matrix` format; the QP kernel converts it once
per subproblem to `SparseMatrixRC<double>` for repeated matrix-vector products.
The chart-dependent energy and its derivative remain explicit element
integrals. The bound-constrained active set and reduced PCG remain in
`boundQuadratic.hpp`; the library's direct linear solvers do not enforce these
inequality constraints. This implementation still uses the full mesh in one
process, including in MPI-enabled builds.

The optional `Source` callback supplies the inverse source metric and its
volume density `sqrt(det(g))`; these must be consistent.
Its default is Euclidean. The fluent example currently uses that default;
custom source metrics can be passed to `ObstacleMapProblem<D>` directly.

Energy, residual, and frozen stiffness all use the same positive symmetric
degree-two library rule, `QF_Simplex<Rd>(2)`: three edge midpoints in 2D and
four interior points in 3D. The old 2D rule used three interior points, so
nonlinear energies and errors can differ after this refactor. For a nonlinear target metric this defines a
quadrature approximation of the continuum energy. The assembled residual is the
exact derivative of that discrete energy. Quadrature order is currently fixed.

The source boundary is fitted to the computational mesh. No elements are cut and
no ghost penalty is needed. The fitted polyhedral ball has planar boundary faces
with vertices on the sphere; it is not an exact curved ball. Both box meshes use
the native structured mesh constructors. The ball maps their cube vertices by
`x_i * sqrt(1 - (x_j^2+x_k^2)/2 + x_j^2*x_k^2/3)`, with cyclic indices.
This smooth map replaces max-norm radial scaling, which can invert cells of the
library's tetrahedral subdivision. Element and boundary measures are refreshed
after moving vertices, and nonpositive cell volumes are rejected. A future CutFEM
source implementation would need consistent boundary imposition and suitable
cut-cell stabilization before these guarantees could be transferred.

## Geometry first, with global coupling retained

An outer iteration performs the following steps.

1. Mark a source element near the collar when one of its vertices has
   `r_i<tube_radius`. Mark every node of that element for the collar blocks.
   This provides an element overlap at the collar boundary.
2. Update the normal coordinates of free collar nodes with a lower-bound
   quadratic subproblem and an energy line search.
3. Reassemble at that map and update tangential coordinates of the collar
   nodes. Reassemble after acceptance.
4. Reclassify the collar and update all target coordinates of free bulk nodes
   together. Repeat for `bulk_sweeps` sweeps, reclassifying and reassembling each
   time.
5. Evaluate both global KKT residuals at the same current map.

Thus the obstacle geometry receives the first updates of each cycle, and the
harmonic bulk is relaxed afterwards. Contact is never permanently frozen.
Classification selects algebraic variables; it does not change the energy or
insert an internal Dirichlet boundary. Every incident element contributes to
each selected variable, including elements crossing the collar threshold.
The full metric is used there even when some vertices lie outside the Fermi
region. Bulk normal coordinates retain their lower bound so that a step cannot
cross the obstacle. Subsequent classification permits contact to reappear.

For a block of selected coordinates, the subproblem is

\[
 \min_d\;g_B^Td+\frac12d^TB_Bd,
 \qquad d_i\geq-r_i\text{ for selected normal coordinates},
\]

with no lower bound on tangential coordinates and zero increments outside the
selected block. Here `B_B` is the selected principal submatrix of the frozen full
metric stiffness plus `sigma*M`, where `M` is the positive lumped source mass.
`sigma` regularizes the step only; it never enters the reported physical energy
or reaction. It makes the quadratic subproblem strictly convex even when a
block has a null mode in its stiffness.

The common lower-bound solver starts feasible, handles active constraints, and
solves on the free variables using preconditioned CG, with coordinate
minimization as a safeguard. Its own projected KKT test determines convergence.
The inner tolerance is tightened using the ratio of lumped mass to the step
matrix diagonal, so mesh refinement cannot hide an unresolved outer residual
behind a small diagonally scaled inner residual.
This kernel is reusable for scalar obstacle subproblems; the map solver does not
require a scalar-obstacle formulation hardcoded for one target.

For an exact subproblem solution, feasibility of the zero increment implies
`g_B^T d <= -d^T B_B d`. A nonzero step therefore gives descent. The implementation
also checks descent for the computed step, then tries `alpha=1,backtrack,...`.
Acceptance requires both complete chart feasibility and

\[
 E_h(q+\alpha d)\leq E_h(q)+\texttt{armijo}\,\alpha g_B^Td.
\]

The floating-point comparison permits an additive allowance of
`64*machine_epsilon*max(1,E_h(q))` to account for energy summation near
stationarity. Thus accepted energy is nonincreasing up to that roundoff
allowance; convergence still requires the global KKT residual and is never
declared from an unresolvable energy change.

Positive nodal normal coordinates remain feasible for `0<=alpha<=1`. Chart
feasibility is checked independently because a coordinate direction can leave
the valid target chart while still satisfying `r>=0`.

## Stopping, reaction, and output

For free nodes only, with lumped mass `M_i`, the implemented residuals are

\[
 R_s=\max_{i,c<m}|g_{i,c}/M_i|,\qquad
 R_r=\max_i\left|r_i-\max(0,r_i-g_{i,r}/M_i)\right|.
\]

Coordinates in these formulas are indexed from one through `m`. The normal
expression is the projected residual with step parameter one. Its zero set is
precisely nonnegative-coordinate KKT complementarity. Residual magnitudes depend
on coordinate and metric scaling; the tolerance is not an invariant physical
error bound. The solver returns `CONVERGED` only when `R_s+R_r<=tolerance`.
Reaching an iteration limit, failure of the inner QP, a non-descent direction,
or an exhausted line search returns a distinct failure status. Invalid geometry
or malformed input raises an exception.

The returned `reaction[i]=g_{i,r}/M_i` is the discrete lumped normal multiplier
at free nodes. At contact it has the physical inward-normal interpretation when
the chart satisfies the Fermi convention. It can contain negative values at
unconverged iterates. Reactions at prescribed nodes are undefined and stored as
NaN; those rows contain Dirichlet forces and must not be interpreted as obstacle
reaction. VTK output uses a separate validity flag.
These are nodal multipliers for the lumped discrete pairing. Interpolating them
as a P1 field can spread the displayed reaction into transition cells; this is
not a pointwise reconstruction of the continuum reaction measure.

`contact_tolerance` labels nodal contact for diagnostics and output. It does not
define the QP active constraints or change the stopping test. A cell is marked
as contact in VTK when all its vertices pass that tolerance. This is a discrete
contact estimate, not a high-order free-boundary reconstruction. Contact need
not have positive reaction: degenerate thin-contact examples can have zero
reaction.

The driver writes `<prefix>.vtk`, `<prefix>_energy.csv`, and
`<prefix>_summary.csv`. Mesh and nodal fields use the library's `Paraview` and
`FunFEM` output at 17-digit precision. Its VTK layout duplicates shared vertices
per cell; the numerical FE mesh still shares its vertex DOFs. VTK ambient
components are nodal samples of `F(q_h)`;
linear interpolation of these displayed samples is not the actual mapped FE
field. The reference error instead evaluates `F(q_h)` at quadrature points. That
driver diagnostic includes `problem.sourceMetric(x).density` in its integration
weights. It measures ambient embedding distance rather than intrinsic target
geodesic distance.

## Reference problems and verification

The mathematical sources are local manuscripts supplied for this task:

- [Some examples of obstacle maps between balls and spherical caps](/home/darth/Documents/PlatonicWorld/Git-Projects/HarmonicObstacleMaps/Main.tex),
  labels `eq:u.ansatz`, `prop:profile.existence`, `thm:global.minimiser`, and
  `thm:veronese.minimiser`.
- [Regularity of obstacle maps between manifolds with boundary](/home/darth/Documents/PlatonicWorld/Git-Projects/ManifoldRegularityConstraintMaps/mfd-constraint-maps.tex),
  labels `eq:energy.density.split`, `eq:normal.eq.strong`,
  `eq:tangent.eq.strong`, `eq:obstacle.ODE`, and `sec:warped.products.II`.

These supply mathematics and verification examples. The solver's discretization
and implementation scope are described above.

### Rotating profile on a three dimensional source

The default is the paper's spherical-cap profile with
`alpha=acos(-0.5)`, `a=0.3`, and `k=1`:

\[
 F(s,r)=(\sin(\alpha-r)\cos s,\sin(\alpha-r)\sin s,\cos(\alpha-r)),
 \quad s=kx_1,\quad r=\alpha-\Phi(x_2).
\]

Here `Phi=alpha` on `|x_2|<=a`. Outside, the even profile solves

\[
 \Phi''=k^2\sin\Phi\cos\Phi,\qquad \Phi(a)=\alpha,\quad\Phi'(a)=0.
\]

The benchmark helper integrates this regular second-order IVP with RK4. It does
not solve the FEM problem to generate its reference. Its metric and contact
reaction are

\[
 G=\operatorname{diag}(\sin^2(\alpha-r),1),\quad
 \partial_rG_{ss}=-2\sin(\alpha-r)\cos(\alpha-r),\quad
 \lambda=-k^2\sin\alpha\cos\alpha\,1_{\{|x_2|<a\}}.
\]

The continuum contact region is a slab and its free boundary consists of the
two planes `x_2=+-a` intersected with the source. The paper proves global
minimality on the unit ball when `k^2(1+h^2)<=pi^2`, where `h=cos(alpha)`.
The selected parameters satisfy this bound strictly. Restricting this minimizer
to the fitted polyhedral subdomain and prescribing its trace preserves
minimality by extending competitors with the exact map. For the optional cube,
the same comparison proof applies since its first Dirichlet eigenvalue is
`3*pi^2/4 > 1.25`. The finite element trace is the nodal interpolation of the
exact coordinates, so interpolation and quadrature errors remain present.

Initial free nodes have strictly positive normal coordinates and a perturbed
tangential field. The initial field does not encode the exact contact slab.
Refinement tests should compare physical-map L2 error, contact localization,
reaction away from the free boundary, energy descent, and the full KKT residual.
Small tube radii should produce both collar and bulk updates.

#### Viewing the map and the full target in ParaView

Open `<prefix>.vtk` and click **Apply**. The file initially displays the source
domain, with the map components stored as point-data arrays.

To display the mapped image for the default spherical-cap example:

1. Select the dataset and apply **Calculator**, with **Attribute Type** set to
   **Point Data**.
2. Enter `u0*iHat + u1*jHat + u2*kHat` as the expression.
3. Check **Coordinate Results**, then click **Apply**. This checkbox is needed
   to move the vertices; otherwise Calculator only creates a vector array.
4. Hide the original dataset, keep the Calculator output visible, and reset
   the camera.
5. Color by `r` or `contact` to distinguish the contact region. ParaView linearly
   interpolates the mapped vertices, so this display approximates the curved
   map `F(q_h)`.

To see the target portions that the map does not reach, add the entire target
as a transparent reference surface. For the default target
`S^2 intersect {z >= -0.5}`:

1. Add a **Sphere** from the **Sources** menu.
2. Set **Center** to `(0, 0, 0)`, **Radius** to `1`, **Start Phi** to `0`, and
   **End Phi** to `120` degrees. The last angle is `acos(-0.5)` in degrees, so
   this creates the admissible spherical cap.
3. Set **Theta Resolution** and **Phi Resolution** to approximately `100`,
   then click **Apply**. These parameters are described in the
   [ParaView Sphere documentation](https://www.paraview.org/paraview-docs/latest/python/paraview.simple.__init__.Sphere.html).
4. Choose **Solid Color** for the cap and set its **Opacity** to about `0.15`.
   Keep the mapped Calculator output visible, colored, and opaque. See
   [ParaView display settings](https://docs.paraview.org/en/latest/UsersGuide/displayingData.html).

The reached region appears in color and the rest of the target remains visible
as a faint surface. This is a visual overlay, not a computed set difference.
For another target geometry, supply its corresponding reference surface instead
of using these spherical-cap settings.

### Exterior-circle profile

This smooth Cartesian verification problem is derived from the warped-product
ODE in the second manuscript; it is not the printed annular Nitsche formula.
On a box take `s=kx_1`, `a=0.3`, `k=1`, and

\[
 F(s,r)=((1+r)\cos s,(1+r)\sin s),\qquad
 r(x)=\begin{cases}0,&|x_2|\leq a,\\
 \cosh(k(|x_2|-a))-1,&|x_2|>a.
 \end{cases}
\]

Then `G=diag((1+r)^2,1)`. Orthogonal source dependencies give
`grad(s).grad(r)=0`, and `r''=k^2(1+r)` off contact verifies both coupled
equations. The reaction is `k^2` on contact. For any admissible competitor `v`,
the half-energy difference satisfies

\[
 E(v)-E(u)\geq\tfrac12(\lambda_1(\Omega)-k^2)
                  \|v-u\|_{L^2(\Omega)}^2.
\]

This follows from `-Delta u=k^2 u` on contact, zero Laplacian elsewhere, and
`u.(v-u)>=-|v-u|^2/2` where `|u|=1` and `|v|>=1`. Thus the selected profile is a
global minimizer on either `[-1,1]^2` or `[-1,1]^3`. Its maximum distance is
`cosh(0.7)-1`, approximately `0.255`; a collar radius `0.1` deliberately leaves
part of its image for bulk harmonic relaxation.

### Tangential reparameterization and general metrics

Set `theta(s)=s+epsilon*sin(s)`, with `epsilon=0.2`. Compose either physical
benchmark with this coordinate change and replace the exact tangential field
by `inverseLongitude(k*x_1,epsilon)`. The physical map is unchanged. If the
original tangential coefficient is `A(r)`, the transformed coefficient is

\[
 \widetilde A(s,r)=A(r)(1+\epsilon\cos s)^2,\qquad
 \partial_s\widetilde A=-2\epsilon A(r)(1+\epsilon\cos s)\sin s.
\]

This tests the tangential metric derivative without adding an artificial
forcing. `reparameterizedChart()` transforms the full matrix and all its first
derivatives, including mixed entries. At finite mesh size the coordinate P1
approximation spaces differ under nonlinear reparameterization, so one should
expect convergence to the same physical map, not identical discrete solutions.

Complement these exact profiles with a directional finite-difference check of
the assembled energy gradient for a full, non-diagonal metric, including
`s-r` cross terms outside the collar. Check preservation of prescribed values,
feasibility between nodes, failure for invalid chart certificates, and monotonic
accepted energy. Higher target dimension and the scalar half-space reduction
exercise the dimension-independent assembly. QP tests should verify actual KKT
conditions, nonzero lower bounds, and release of active constraints. Numerical
results from the implementation are recorded below.

### Verified results

All three library-linked QP, map, and embedding-geometry suites pass through CTest.
Tests check the derivative
of the assembled energy against independent centered differences, including
target metric cross terms, tangential derivatives, a variable source metric,
2D/3D source meshes, and target dimensions 1, 3, and 4. They also check prescribed
values, feasibility, multiplier signs, contact away from the interface,
non-Fermi bulk relaxation, and explicit failure statuses.
The embedding suite also solves a contact problem above a concave parabola,
checks its positive reaction, and verifies a coupled metric for a curved target
embedded in four dimensions.

After the library refactor, the spherical-cap verification profile (`k=1`,
`tube=0.01`, tolerance `1e-7`) on fitted polyhedral balls gives:

| Subdivisions | Nodes | Tetrahedra | Outer iterations | Global residual | Physical-map L2 error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 125 | 384 | 6 | 1.07e-8 | 8.87924e-3 |
| 6 | 343 | 1296 | 7 | 4.73e-8 | 5.50908e-3 |

The example in `mainFiles/obstacle_map.cpp` currently uses `k=2.712423`.
At 8 subdivisions with its default tube it converges in 19 iterations, with
residual `5.33e-8` and physical-map L2 error `1.74062e-2`.

The independent exterior-circle test on square meshes, with tube radius `0.01`,
reduces L2 error from `1.10545e-2` at 8 subdivisions to `2.39564e-3` at 16.
Both use bulk harmonic updates and converge below `1e-7`. Reparameterizing its
tangential coordinate gives L2 error `1.10471e-2` on the coarse mesh, consistent
with recovery of the same physical map from a different coordinate FE space.
All these runs preserve nodal feasibility; dual violations in the profile tests
are within the residual tolerance. These are numerical verification results,
not a general convergence theorem or a certified free-boundary error bound.

## Build and run

With an existing configured project build, the example target is `obstacle_map`.
The verification targets are `obstacle_map_tests`, `bound_quadratic_tests`, and
`obstacle_target_tests`,
enabled by the separate option `CUTFEM_BUILD_OBSTACLE_MAP_TESTS=ON`.

```sh
cmake -S . -B build -DCUTFEM_BUILD_OBSTACLE_MAP_TESTS=ON
cmake --build build --target obstacle_map
build/bin/obstacle_map --subdivisions 8 --tube 0.03 --output /tmp/obstacle_map
cmake --build build --target obstacle_map_tests bound_quadratic_tests obstacle_target_tests
ctest --test-dir build --output-on-failure
```

The example and tests now link `FESpace` and `common`; direct compilation of a
standalone header is no longer supported. MPI and external direct solvers are
optional for these targets. A minimal serial configuration is:

```sh
cmake -S . -B build/obstacle-serial -DUSE_MPI=OFF -DUSE_MUMPS=OFF -DUSE_UMFPACK=OFF -DCUTFEM_CREATE_DOCS=OFF -DCUTFEM_BUILD_OBSTACLE_MAP_TESTS=ON
cmake --build build/obstacle-serial --target obstacle_map obstacle_map_tests bound_quadratic_tests obstacle_target_tests
ctest --test-dir build/obstacle-serial --output-on-failure
```

The example accepts `--subdivisions N`, `--box`, `--tube R`, `--tol T`,
`--max-iterations N`, and `--output PREFIX`. Without `--box`, it uses the fitted
polyhedral ball. The solver returns exit code zero only on convergence, one on
nonconvergence, and two on invalid input or another caught exception.

## Current limits and next extensions

The nonlinear energy is generally nonconvex. A small global KKT residual
certifies stationarity of the quadrature-defined constrained problem within the
chosen coordinate representation. It does not certify a global minimizer or a
correct topological class for arbitrary boundary data. The exact examples have
separate minimality arguments. Energy descent alone is not a proof that every
nonlinear run will converge before its iteration or line-search limits.

General atlas transitions, periodic angle degrees of freedom, nontrivial
winding, automatic chart construction from a level set, and changes of target
topological class are not implemented. Elements outside the provider's
certified parameter domain are rejected. Problem setup must also reject
boundary data that require an unavailable atlas or a nonexistent global lift;
the solver has no independent winding-number check. In particular, nodal `atan2`
interpolation across a branch cut does not implement the manuscript's degree-one
annulus example. Permitting arbitrary real longitude in a covering
parameterization does not supply a global lift for that winding map. The
spherical parameterization excludes the pole and does not cover the entire cap.
A target representation that becomes singular or ceases to cover the image needs a
new representation; decreasing the optimization step is not a substitute for
an atlas.

The source implementation has no dimensions above three, no MPI distribution,
adaptive refinement, higher order coordinate elements, curved source elements,
or certified free-boundary error estimator. The manuscript's Veronese example
has source dimension at least twelve and a discontinuity at thin contact, so it
is an important limitation of a smooth single-chart method, not a supported
benchmark here. The planar radial map from a full disk to its exterior has
divergent energy at the origin and is not a substitute for the annular test.

Useful extensions are an atlas or manifold finite element representation,
adaptive refinement driven by the free boundary and harmonic error, higher
order quadrature, and scalable sparse solvers. A future staged free-boundary
acceleration must still re-evaluate the complete coupled KKT system after bulk
relaxation and permit contact changes.
