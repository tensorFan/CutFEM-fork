# A round free boundary in hyperbolic geometry

This example seeks elegance through a simple geometric mechanism: a map from
the Euclidean unit ball into hyperbolic four-space touches a horosphere on a
smaller concentric ball, then leaves it quadratically. Its free boundary is an
exact round sphere. The construction has a nonzero obstacle reaction and an
energy comparison proving global minimality and uniqueness, including against
competitors without radial symmetry.

![Computed free-boundary proxy, equatorial section, and radial profiles](finest.png)

The gold sphere in the figure is the mathematical reference. The teal surface
is extracted from the computed tetrahedral field; it is a numerical proxy for
the free boundary. No surface smoothing is applied.

## Source, target, and energy

The source is the unit ball \(B^3\subset\mathbb R^3\), with its Euclidean metric.
The target is hyperbolic four-space of sectional curvature \(-\kappa^2\), in
global horospherical coordinates \(q=(s,r)\in\mathbb R^3\times\mathbb R\):

\[
G=dr^2+e^{2\kappa r}|ds|^2,\qquad \kappa>0.
\]

The obstacle restricts the map to \(r\geq0\), the exterior of a horoball. The
boundary \(r=0\) is a horosphere. These are global Fermi coordinates: the
vertical curves are unit-speed normal geodesics, and \(r\) is exactly the
distance to the obstacle throughout the admissible target.

To identify this with the upper-half-space model, set
\(z=e^{-\kappa r}/\kappa\). Then

\[
G=\frac{|ds|^2+dz^2}{(\kappa z)^2},\qquad 0<z\leq1/\kappa.
\]

For the experiment \(\kappa=1\), this becomes \(z=e^{-r}\). The callback
returning \((s,z)\) supplies upper-half-space coordinates, not an isometric
Euclidean embedding. The target metric in the solver remains hyperbolic.

With a prescribed trace, minimize the half Dirichlet energy

\[
E(q)=\frac12\int_{B^3}
\left(|\nabla r|^2+e^{2\kappa r}\sum_{j=1}^3|\nabla s_j|^2\right)dx,
\qquad r\geq0.
\]

## A radial construction

Write \(\rho=|x|\), and seek

\[
u(x)=\left(f(\rho)\frac{x}{\rho},R(\rho)\right).
\]

Choose a contact radius \(a\) and a tangential scale \(c\). On \(0\leq\rho\leq a\),
set \(f=c\rho\), \(R=0\), so \(u=(cx,0)\), including at the origin. Outside
contact, solve the initial-value problem

\[
\begin{aligned}
f''&=\frac{2f}{\rho^2}-\left(\frac2\rho+2\kappa R'\right)f',\\
R''&=\kappa e^{2\kappa R}
\left(f'^2+\frac{2f^2}{\rho^2}\right)-\frac{2R'}\rho,
\end{aligned}
\qquad
(f,f',R,R')(a)=(ca,c,0,0).
\]

These are the full harmonic-map equations for the radial ansatz. Thus the
constructed map is harmonic on the noncontact region, rather than merely
stationary within a reduced scalar problem. Prescribe its trace on the unit
sphere:

\[
u|_{\partial B^3}=(Fx,b),\qquad F=f(1),\quad b=R(1).
\]

The experiment uses

\[
a=\frac12,\qquad c=\frac1{\sqrt3},\qquad\kappa=1.
\]

The constants \(F,b\) are defined by the exact IVP. Independent RK4 integration
with 4096 steps, as implemented in
[horosphere.hpp](../../../cpp/example/obstacle_map/horosphere.hpp), gives

\[
F\approx0.564569261944883,\qquad
b\approx0.084453987587878.
\]

The continuum energy is

\[
E(u)=2\pi\left[c^2a^3+
\int_a^1\left(e^{2\kappa R}(\rho^2f'^2+2f^2)+\rho^2R'^2\right)d\rho\right]
\approx2.23125954799122.
\]

RK4 integration and Simpson quadrature with 4096, 16384, and 32768 steps
agree within \(3\times10^{-14}\) for these values. This is the reference energy
on the exact unit ball; the finite element energy is evaluated on the fitted
polyhedral source.

### Existence through the outer boundary

There is an elementary bound ensuring the IVP exists up to \(\rho=1\). The
normal equation gives

\[
(\rho^2R')'=\kappa\rho^2e^{2\kappa R}
\left(f'^2+2f^2/\rho^2\right)>0.
\]

Hence \(R'>0\) outside contact. The invariant region
\(f>0\), \(0<f'\leq c\), \(f\leq c\rho\) follows directly from the tangential
equation: at \(f'=0\) its derivative is positive; at \(f'=c\), with
\(f\leq c\rho\), its derivative is nonpositive; and
\((f-c\rho)'\leq0\). Consequently

\[
R''\leq3\kappa c^2e^{2\kappa R}.
\]

Comparison with the solution having the same zero initial value and slope,

\[
Y(\rho)=-\frac1\kappa
\log\cos\bigl(\sqrt3\kappa c(\rho-a)\bigr),
\]

gives \(R\leq Y\) and \(R'\leq Y'\). Both remain bounded if
\(\sqrt3\kappa c(1-a)<\pi/2\). Here the argument is exactly
\(1/2\), and \(Y(1)=-\log\cos(1/2)\approx0.13058424\). All four IVP variables remain
bounded, so the solution continues to the outer boundary. This existence
argument does not depend on the numerical integration.

### The free boundary and its contact order

Since \(R=0\) inside and \(R'>0\) outside, the source free boundary is exactly

\[
\Gamma=\{|x|=a\}.
\]

Its area is \(4\pi a^2=\pi\), and the enclosed contact volume is \(\pi/6\).
Its image lies on the horosphere
\(r=0\), where the tangential metric is Euclidean: it is a round sphere of
radius \(ca=1/(2\sqrt3)\approx0.28867513\) in that horosphere.

The contact reaction is constant:

\[
\lambda=3\kappa c^2=1.
\]

The normal equation at release gives \(R''(a+)=3\kappa c^2\). Therefore

\[
R(\rho)=\frac{3\kappa c^2}{2}(\rho-a)^2
+O\bigl((\rho-a)^3\bigr),\qquad\rho\downarrow a\text{ from above}.
\]

The normal distance leaves the obstacle quadratically, with coefficient
\(1/2\). The map is \(C^1\), and locally \(C^{1,1}\), across the free boundary;
it is not \(C^2\), because the normal second derivative jumps from zero to
\(1\). This nonzero reaction and quadratic release make the sphere a
nondegenerate contact interface.

## Global minimality and uniqueness

Let \(v\) be any feasible competitor with the same trace. In the complete
ambient hyperbolic space, put
\(\xi=\log_u v\) and \(d(x)=d_{\mathbb H^4}(u(x),v(x))\). The pointwise geodesic
homotopy is allowed to leave \(r\geq0\); it is used to compare energies in the
ambient space. Nonpositive curvature gives

\[
E(v)-E(u)\geq DE(u)[\xi]+\frac12\int_{B^3}|\nabla d|^2.
\]

This follows from the second-variation formula and the inequality
\(|\nabla|\partial_tu_t||\leq|\nabla\partial_tu_t|\). The corresponding Sobolev
energy-convexity statement appears in
[Mese, *Uniqueness theorems for harmonic maps into metric spaces*, pp. 2–3](https://math.jhu.edu/~cmese/uniqueness.pdf).
That reference uses the energy without the factor \(1/2\).

The constructed map has tension \(-\lambda\partial_r\) on the contact ball
\(C=B_a\), and zero tension outside. Its first derivatives match at \(a\), so
there is no interface distribution in the weak first variation:

\[
DE(u)[\xi]=\lambda\int_C dr(\xi).
\]

The global coordinate function \(r\) satisfies

\[
\operatorname{Hess}r=\kappa(G-dr\otimes dr)\leq\kappa G.
\]

Taylor's formula along the ambient geodesic consequently gives
\(dr(\xi)\geq r(v)-r(u)-\kappa d^2/2\). On contact, \(r(u)=0\) and feasibility
gives \(r(v)\geq0\). Combining the estimates,

\[
\begin{aligned}
E(v)-E(u)
&\geq\frac12\int_{B^3}|\nabla d|^2
-\frac{3\kappa^2c^2}{2}\int_C d^2\\
&\geq\frac12\bigl(\lambda_1(B^3)-3\kappa^2c^2\bigr)
\int_{B^3}d^2.
\end{aligned}
\]

Here \(d\) has zero boundary trace, and the first Dirichlet eigenvalue of the
Euclidean unit three-ball is \(\lambda_1(B^3)=\pi^2\). For the chosen parameters,

\[
E(v)-E(u)\geq
\frac{\pi^2-1}{2}\int_{B^3}d^2
\approx4.43480220\int_{B^3}d^2.
\]

Thus \(u\) is the unique global minimizer. In particular, no symmetry assumption
on competitors is required. More generally, the same conclusion holds for
this construction whenever the IVP exists and \(3\kappa^2c^2<\pi^2\).

The argument extends to finite-energy Sobolev competitors. First truncate
\(r\) from above, which preserves feasibility and decreases its contribution
to the tangential metric weight. Then project \(s\) onto increasing Euclidean
balls. Choose both bounds above the prescribed boundary values. For bounded
competitors, \(\log_u v\) is an admissible Sobolev test field for the weak first
variation, and geodesic energy convexity applies. Passing successively to the
limits gives convergence of energy and of the intrinsic \(L^2\) distance, and
retains the displayed inequality.

## What the finite element experiment verifies

The executable uses the existing obstacle-map solver on a fitted polyhedral
ball \(P\subset B^3\), with tetrahedral P1 coordinates and the metric above.
Its initial normal coordinate is strictly positive everywhere, and its
tangential coordinates have a small nonsymmetric perturbation. The radial
contact region is not supplied as an initial active set. The independently
integrated profile defines the boundary data and provides a reference for
diagnostics.

The fitted source is generated by
[horosphere_mesh.hpp](../../../cpp/example/obstacle_map/horosphere_mesh.hpp).
It maps an octant-aligned tetrahedral cube by
\(x\mapsto\|x\|_\infty x/\|x\|_2\), with the origin mapped to itself.
The tetrahedral faces align with the derivative jumps of this map, and all
boundary vertices lie on the unit sphere. This avoids the inverted corner
tetrahedra encountered when refining the earlier smooth cube-to-ball map.
The mesh generator has no contact-radius argument; its uniform radial layers
coincide with \(a=1/2\) when the subdivision count is a multiple of four.

With the exact trace \(u|_{\partial P}\), the continuum proof remains valid on
\(P\): Dirichlet domain monotonicity gives \(\lambda_1(P)\geq\pi^2\). In the
actual P1 computation, boundary nodal values are prescribed from the profile,
then interpolated across boundary facets. The resulting facet trace differs
from the exact continuum trace; a zero `trace_error` checks boundary nodes
only. The computed KKT residual belongs to the quadrature-defined finite
element objective. Its convergence demonstrates discrete stationarity to the
reported tolerance, and does not certify the exact continuum solution or
global minimality of the quadrature problem.

The boundary of the union of fully contacting tetrahedra is extracted directly
for the main figure and surface VTK, provided it passes a closed-manifold
check. It approximates the continuum free boundary. Positive levels
\(r=\varepsilon\) and the midpoint of a nodal contact indicator provide
additional surface diagnostics. Even for the exact solution,
a positive level has radius greater than \(a\), with leading offset
\(\sqrt{2\varepsilon/(3\kappa c^2)}\). The plotting tool reports errors both to
the actual free boundary and to the corresponding reference level. Its
`coordinate_l2` comparison measures the field in the global \((s,r)\)
coordinates, rather than intrinsic hyperbolic distance.

<!-- NUMERICAL_RESULTS -->

## Reproduction and outputs

From the repository root, use the existing configured exploration build:

```sh
cmake --build build/obstacle-explore --target obstacle_map_horosphere obstacle_horosphere_tests -j2
ctest --test-dir build/obstacle-explore -R obstacle_horosphere --output-on-failure
build/obstacle-explore/bin/obstacle_map_horosphere --subdivisions 20 --radius .5 --slope 0.5773502691896258 --seed 0 --tol 1e-7 --output build/obstacle-explore/runs/horosphere20
python3 tools/plot_obstacle_horosphere.py build/obstacle-explore/runs/horosphere20 --pdf
```

Change `--subdivisions` to refine the fitted ball (odd counts round up to the
next even count), and `--seed` to change the
initial perturbation. The executable's `--help` lists its solver controls.
NumPy and Matplotlib are required by the plotting tool.

For a run prefix `PREFIX`, the solver writes:

- `PREFIX_nodes.csv`: source points, computed target coordinates, reactions,
  boundary flags, and reference nodal values.
- `PREFIX_cells.csv`: tetrahedral connectivity.
- `PREFIX_profile.csv`: the independently integrated radial reference.
- `PREFIX_summary.csv`: convergence, energy, feasibility, complementarity,
  coordinate errors, and contact diagnostics.
- `PREFIX.vtk` and `PREFIX_energy.csv`: full field and iteration history.

[plot_obstacle_horosphere.py](../../../tools/plot_obstacle_horosphere.py)
extracts surfaces directly from the tetrahedra and adds `PREFIX_surface.vtk`,
`PREFIX_geometry.json`, and `PREFIX_figure.png`; `--pdf` also writes the figure
as a PDF. The geometry JSON records connected components, topology, and
area-weighted radial errors. The mathematical reference requires neither
surface extraction nor a chosen numerical threshold.
