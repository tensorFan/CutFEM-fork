#include "../example/obstacle_map/driver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace cutfem::obstacle;
namespace examples = cutfem::obstacle::examples;

namespace {
using cutfem::obstacle::Vector;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function> void requireInvalid(Function operation, const std::string& message) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

template <int D> void checkResult(const ObstacleMapProblem<D>& problem,
                                  const ObstacleMapResult& result,
                                  const typename ObstacleMapProblem<D>::Data& boundary,
                                  double tolerance = 1e-7) {
    require(result.converged && result.status == "CONVERGED",
            "Map solve failed: " + result.status + ", residual " +
            std::to_string(result.tangent_residual + result.normal_residual));
    require(problem.feasible(result.coordinates), "Returned map is infeasible");
    require(result.tangent_residual + result.normal_residual <= tolerance,
            "Returned map violates stopping tolerance");
    const auto assembly = problem.assemble(result.coordinates);
    const int m = problem.target().dimension;
    double normal = 0., tangent = 0.;
    for (std::size_t i = 0; i < problem.mesh().nv; ++i) {
        if (problem.isBoundary(i)) {
            const auto trace = boundary(point<D>(problem.mesh()(i)));
            require(std::isnan(result.reaction[i]), "Dirichlet reaction was reported as an obstacle multiplier");
            for (int a = 0; a < m; ++a)
                require(result.coordinates[i*m+a] == trace[a], "Dirichlet data changed");
            continue;
        }
        const double r = result.coordinates[i*m+m-1];
        const double lambda = assembly.gradient[i*m+m-1] / assembly.mass[i];
        normal = std::max(normal, std::abs(std::min(r, lambda)));
        require(std::abs(result.reaction[i] - lambda) < 1e-12, "Reaction was not recomputed at the final map");
        for (int a = 0; a < m-1; ++a)
            tangent = std::max(tangent, std::abs(assembly.gradient[i*m+a] / assembly.mass[i]));
    }
    require(std::abs(normal - result.normal_residual) < 1e-12 &&
            std::abs(tangent - result.tangent_residual) < 1e-12, "Residuals refer to different maps");
    for (std::size_t i = 1; i < result.energy_history.size(); ++i)
        require(result.energy_history[i] <= result.energy_history[i-1] +
                1e-12 * (1. + std::abs(result.energy_history[i-1])), "Accepted energy increased");
}

// Euclidean half-space expressed in a nonlinear chart. F(s,t,r) =
// (s + epsilon*max(r-r0,0)^4, t, r). The chart is exactly Fermi in
// r <= r0 and develops tangent/normal metric cross terms in the far field.
TargetChart farFieldShear(double epsilon = .2, double r0 = .05) {
    auto target = examples::euclidean_half_space(3, r0);
    target.embed = [epsilon, r0](const Vector& q) {
        const double h = std::max(q[2] - r0, 0.);
        return Vector{q[0] + epsilon*h*h*h*h, q[1], q[2]};
    };
    target.metric = [epsilon, r0](const Vector& q, Vector& G, std::vector<Vector>& derivative) {
        const double h = std::max(q[2] - r0, 0.);
        const double b = 4.*epsilon*h*h*h, bp = 12.*epsilon*h*h;
        G = {1., 0., b, 0., 1., 0., b, 0., 1.+b*b};
        derivative.assign(3, Vector(9, 0.));
        derivative[2][2] = derivative[2][6] = bp;
        derivative[2][8] = 2.*b*bp;
    };
    return target;
}

template <int D> void gradientCheck(std::shared_ptr<SourceMesh<D>> mesh) {
    // Compose two independent changes of coordinates to exercise derivatives
    // in both a tangential coordinate and the normal coordinate.
    const auto target = examples::reparameterizedChart(farFieldShear(), .23);
    auto source = [](const Point<D>& x) {
        SourceMetric<D> metric;
        metric.inverse[0][0] = 1.2 + .1*x[0];
        metric.inverse[1][1] = 1.4 + .1*x[1];
        metric.inverse[0][1] = metric.inverse[1][0] = .17;
        metric.density = 1. / std::sqrt(metric.inverse[0][0]*metric.inverse[1][1] - .17*.17);
        return metric;
    };
    const ObstacleMapProblem<D> problem(std::move(mesh), target, source);
    const Vector q = problem.interpolate([](const Point<D>& x) {
        return Vector{.15 + .31*x[0] - .17*x[1], -.2*x[0] + .21*x[1], .75 + .12*x[0] + .08*x[1]};
    });
    const auto assembly = problem.assemble(q);
    Vector G; std::vector<Vector> derivatives;
    target.metric({.3, .2, .8}, G, derivatives);
    require(std::abs(G[2]) > .05 && std::abs(derivatives[0][0]) > .01,
            "Gradient fixture lost its metric coupling or tangential derivatives");
    double largest_error = 0.;
    for (std::size_t i = 0; i < q.size(); ++i) {
        Vector plus = q, minus = q;
        const double h = 2e-6;
        plus[i] += h; minus[i] -= h;
        const double numerical = (problem.assemble(plus, false).energy -
                                  problem.assemble(minus, false).energy) / (2.*h);
        largest_error = std::max(largest_error, std::abs(numerical - assembly.gradient[i]));
        require(std::abs(numerical - assembly.gradient[i]) < 2e-8 * (1.+std::abs(numerical)),
                "Full metric gradient does not differentiate the discrete energy");
    }
    std::cout << "gradient D=" << D << " max error=" << largest_error << '\n';
}

// Exact affine energy and volume catch basis, DOF, Jacobian and quadrature
// integration mistakes independently of the nonlinear solver's residuals.
template <int D> void libraryAssembly(std::shared_ptr<SourceMesh<D>> mesh) {
    const ObstacleMapProblem<D> problem(mesh, examples::euclidean_half_space(1));
    const auto q = problem.interpolate([](const Point<D>& x) {
        double value = 2.;
        for (int d = 0; d < D; ++d) value += .1*(d+1)*x[d];
        return Vector{value};
    });
    const auto assembly = problem.assemble(q);
    double volume = 0., norm2 = 0.;
    for (double mass : assembly.mass) volume += mass;
    for (int d = 0; d < D; ++d) norm2 += .01*(d+1)*(d+1);
    require(std::abs(volume-std::pow(2.,D)) < 1e-12, "Library quadrature has wrong volume normalization");
    require(std::abs(assembly.energy-.5*volume*norm2) < 1e-12, "Library P1 affine energy is incorrect");
    Rn product(q.size());
    multiply(q.size(), q.size(), assembly.stiffness, q, product);
    for (std::size_t i = 0; i < q.size(); ++i)
        require(std::abs(product[int(i)]-assembly.gradient[i]) < 1e-12, "Library matrix and residual DOFs disagree");
}

void fittedBallMeshes() {
    for (int n : {2,3,4,6,8,12}) {
        const auto mesh = examples::unit_ball_mesh_3d(n);
        const ObstacleMapProblem<3> problem(mesh, examples::euclidean_half_space(1));
        for (int i = 0; i < mesh->nv; ++i) {
            const double radius = (*mesh)(i).norme();
            require(problem.isBoundary(i) ? std::abs(radius-1.) < 1e-12 : radius < 1.,
                    "Mapped library mesh has incorrect sphere boundary or interior vertices");
        }
    }
}

void halfSpaceSolutions() {
    const auto mesh = examples::box_mesh_2d(4);
    const ObstacleMapProblem<2> problem(mesh, examples::euclidean_half_space(4));
    const auto exact = [](const Point<2>& x) {
        return Vector{.2*x[0] - .1*x[1], -.3*x[0] + .2, .4*x[1], .7 + .1*x[0] - .1*x[1]};
    };
    const auto initial = [exact](const Point<2>& x) {
        auto q = exact(x);
        for (std::size_t a = 0; a < q.size(); ++a) q[a] += .08*std::cos(x[0] + .3*x[1] + a);
        return q;
    };
    const auto trace_only = [exact](const Point<2>& x) {
        require(std::abs(x[0]) == 1. || std::abs(x[1]) == 1., "Dirichlet callback evaluated in source interior");
        return exact(x);
    };
    const auto result = problem.solve(trace_only, initial);
    checkResult<2>(problem, result, trace_only);
    require(examples::referenceL2<2>(problem, result.coordinates, exact) < 2e-8,
            "Affine map into target dimension four is inaccurate");
    require(result.harmonic_steps > 0, "Noncontact affine problem did not use the harmonic block");
    for (bool contact : result.contact) require(!contact, "Positive affine map acquired contact nodes");

    const ObstacleMapProblem<2> contact_problem(mesh, examples::euclidean_half_space(3, .2));
    const auto constant = [](const Point<2>&) { return Vector{.3, -.2, 0.}; };
    const auto positive = [](const Point<2>& x) { return Vector{.3+.1*x[0], -.2+.1*x[1], .1}; };
    const auto contact = contact_problem.solve(constant, positive);
    checkResult<2>(contact_problem, contact, constant);
    require(examples::referenceL2<2>(contact_problem, contact.coordinates, constant) < 3e-7,
            "Constant contact map is inaccurate");
    for (bool flag : contact.contact) require(flag, "All-contact problem left noncontact nodes");
    require(contact.normal_steps > 0 && contact.tangent_steps > 0, "Collar splitting was not exercised");

    const ObstacleMapProblem<2> scalar(mesh, examples::euclidean_half_space(1));
    const auto scalar_exact = [](const Point<2>&) { return Vector{.4}; };
    const auto scalar_result = scalar.solve(scalar_exact, [](const Point<2>&) { return Vector{.1}; });
    checkResult<2>(scalar, scalar_result, scalar_exact);
}

void nonFermiFarFieldSolve() {
    const auto chart = farFieldShear(.2, .05);
    const ObstacleMapProblem<2> problem(examples::box_mesh_2d(6), chart);
    // Constant r makes this ambient affine map exactly representable even in
    // the sheared coordinates; all metric cross terms are nevertheless active.
    const auto exact = [](const Point<2>& x) { return Vector{.3*x[0] - .2*x[1], .1*x[0]+.4*x[1], .8}; };
    const auto initial = [exact](const Point<2>& x) {
        auto q = exact(x);
        q[0] += .06*std::cos(.7*x[0]); q[1] += .04*std::sin(x[1]+.5); q[2] += .05;
        return q;
    };
    const auto result = problem.solve(exact, initial);
    checkResult<2>(problem, result, exact);
    require(result.harmonic_steps > 0 && result.normal_steps == 0 && result.tangent_steps == 0,
            "Far-field map unexpectedly depended on collar splitting");
    require(examples::referenceL2<2>(problem, result.coordinates, exact) < 3e-8,
            "Non-Fermi far-field metric did not recover the affine harmonic map");
}

void rejectionAndFailure() {
    auto mesh = examples::box_mesh_2d(2);
    const ObstacleMapProblem<2> problem(mesh, examples::euclidean_half_space(2));
    auto q = problem.interpolate([](const Point<2>& x) { return Vector{x[0], .1}; });
    q[1] = -.01;
    require(!problem.feasible(q), "Negative normal coordinate accepted");
    requireInvalid([&] { problem.assemble(q); }, "Infeasible assembly was accepted");

    auto certificate = examples::euclidean_half_space(2);
    certificate.admissible_simplex = [](const std::vector<Vector>& vertices) {
        double lo = vertices.front()[0], hi = lo;
        for (const auto& value : vertices) { lo = std::min(lo, value[0]); hi = std::max(hi, value[0]); }
        return hi-lo < .5;
    };
    const ObstacleMapProblem<2> limited(mesh, certificate);
    const auto varying = limited.interpolate([](const Point<2>& x) { return Vector{x[0], .1}; });
    require(!limited.feasible(varying), "Whole-simplex geometry certificate was ignored");

    auto bad_metric = examples::euclidean_half_space(2);
    bad_metric.metric = [](const Vector&, Vector& G, std::vector<Vector>& dG) {
        G = {1., 2., 2., 1.}; dG.assign(2, Vector(4, 0.));
    };
    const ObstacleMapProblem<2> indefinite(mesh, bad_metric);
    const auto feasible = problem.interpolate([](const Point<2>&) { return Vector{0., .1}; });
    requireInvalid([&] { indefinite.assemble(feasible); }, "Indefinite target metric was accepted");
    const ObstacleMapProblem<2> bad_source(mesh, examples::euclidean_half_space(2), [](const Point<2>&) {
        SourceMetric<2> source; source.inverse[0][0] = -1.; return source;
    });
    requireInvalid([&] { bad_source.assemble(feasible); }, "Indefinite source metric was accepted");
    mesh = examples::box_mesh_2d(2);
    static_cast<R2&>(mesh->v((*mesh)(0,1))) = mesh->v((*mesh)(0,0));
    requireInvalid([&] { ObstacleMapProblem<2> degenerate(mesh, examples::euclidean_half_space(2)); },
                   "Degenerate source simplex was accepted");

    const auto zero = [](const Point<2>&) { return Vector{0., 0.}; };
    const auto initial = [](const Point<2>&) { return Vector{.1, .1}; };
    ObstacleMapOptions options;
    options.max_iterations = 0;
    const auto stopped = problem.solve(zero, initial, options);
    require(!stopped.converged && stopped.status == "MAX_ITERATIONS", "Iteration limit gave false convergence");
    options.sigma = 0.;
    requireInvalid([&] { problem.solve(zero, initial, options); }, "Invalid solver option was accepted");

    // A deliberately conservative certificate accepts the initial simplex
    // hulls but refuses every changed hull. This exercises exhausted geometry
    // backtracking without relying on a particular floating-point iteration.
    auto refusing = examples::euclidean_half_space(2, .2);
    refusing.admissible_simplex = [](const std::vector<Vector>& vertices) {
        for (const auto& value : vertices) if (value[1] != 0. && value[1] != .1) return false;
        return true;
    };
    const ObstacleMapProblem<2> restricted(examples::box_mesh_2d(2), refusing);
    options = {}; options.max_backtracks = 3;
    const auto rejected_step = restricted.solve(zero, initial, options);
    require(!rejected_step.converged && rejected_step.status == "LINE_SEARCH_FAILED",
            "Exhausted geometry backtracking did not return LINE_SEARCH_FAILED");
}

template <int D, class Profile> double profileError(std::shared_ptr<SourceMesh<D>> mesh, const Profile& profile,
                                                   double tube, int subdivisions, const std::string& label) {
    const ObstacleMapProblem<D> problem(std::move(mesh), profile.chart(tube));
    const auto exact = [profile](const Point<D>& x) { return profile.exact(x); };
    const auto initial = [profile](const Point<D>& x) { return profile.initial(x, false); };
    const auto result = problem.solve(exact, initial);
    std::cout << label << " status=" << result.status << " iterations=" << result.iterations
              << " residual=" << result.tangent_residual+result.normal_residual << std::endl;
    checkResult<D>(problem, result, exact);
    bool contact = false, noncontact = false;
    int slab_checks = 0, far_checks = 0;
    double primal_violation = 0., dual_violation = 0.;
    const double h = 2. / subdivisions, interface_strip = .35*h;
    for (std::size_t i = 0; i < problem.mesh().nv; ++i) if (!problem.isBoundary(i)) {
        const auto x = point<D>(problem.mesh()(i));
        const double r = result.coordinates[2*i+1], lambda = result.reaction[i];
        require(initial(x).back() > 0., "Profile initial map supplied a contact set");
        contact = contact || result.contact[i]; noncontact = noncontact || !result.contact[i];
        primal_violation = std::max(primal_violation, std::max(0., -r));
        dual_violation = std::max(dual_violation, std::max(0., -lambda));
        double domain_margin;
        if constexpr (D == 3) domain_margin = 1. - std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        else domain_margin = 1. - std::max(std::abs(x[0]), std::abs(x[1]));
        if (domain_margin < .3*h) continue;
        if (std::abs(x[1]) < profile.a - interface_strip) {
            ++slab_checks;
            require(r < 2e-7 && lambda > .1*profile.reaction(x),
                    "Interior contact slab has wrong distance or reaction sign");
        }
        if (std::abs(x[1]) > profile.a + interface_strip) {
            ++far_checks;
            require(r > 1e-7 && std::abs(lambda) < 2e-7,
                    "Interior noncontact region has wrong distance or nonzero reaction");
        }
    }
    require(contact && noncontact, "Profile did not identify both contact and noncontact nodes");
    require(slab_checks > 0 && far_checks > 0, "Profile mesh supplied no independent slab/far-field checks");
    require(primal_violation == 0. && dual_violation < 2e-7, "Profile violates primal/dual feasibility");
    if (D == 2) require(result.harmonic_steps > 0, "Exterior-circle profile did not use the far-field harmonic block");
    const double error = examples::referenceL2<D>(problem, result.coordinates, exact);
    std::cout << label << " L2=" << error << " harmonic steps=" << result.harmonic_steps
              << " max primal violation=" << primal_violation << " max dual violation=" << dual_violation << std::endl;
    return error;
}

void profileRefinement() {
    const examples::RotatingProfile rotating;
    const double rotating_coarse = profileError<3>(examples::unit_ball_mesh_3d(4), rotating, .01, 4, "rotating n=4");
    const double rotating_fine = profileError<3>(examples::unit_ball_mesh_3d(6), rotating, .01, 6, "rotating n=6");
    require(rotating_fine < .85*rotating_coarse, "Rotating-profile error did not decrease under refinement");
    const examples::ExteriorCircleProfile circle;
    const double circle_coarse = profileError<2>(examples::box_mesh_2d(8), circle, .01, 8, "exterior circle n=8");
    const double circle_fine = profileError<2>(examples::box_mesh_2d(16), circle, .01, 16, "exterior circle n=16");
    require(circle_fine < .85*circle_coarse, "Exterior-circle error did not decrease under refinement");

    const ObstacleMapProblem<2> reparameterized(examples::box_mesh_2d(8),
                                               examples::reparameterizedChart(circle.chart(.01), .2));
    const auto exact = [circle](const Point<2>& x) {
        auto q = circle.exact(x); q[0] = examples::inverseLongitude(q[0], .2); return q;
    };
    const auto initial = [circle](const Point<2>& x) {
        auto q = circle.initial(x, false); q[0] = examples::inverseLongitude(q[0], .2); return q;
    };
    const auto result = reparameterized.solve(exact, initial);
    checkResult<2>(reparameterized, result, exact);
    const double reparameterized_error = examples::referenceL2<2>(reparameterized, result.coordinates, exact);
    require(reparameterized_error < 1.5*circle_coarse, "Reparameterized chart has unexpectedly large physical error");
    std::cout << "reparameterized circle n=8 L2=" << reparameterized_error << std::endl;
}

} // namespace

int main() {
    try {
        gradientCheck<2>(examples::box_mesh_2d(2));
        gradientCheck<3>(examples::box_mesh_3d(1));
        libraryAssembly<2>(examples::box_mesh_2d(2));
        libraryAssembly<3>(examples::box_mesh_3d(2));
        fittedBallMeshes();
        halfSpaceSolutions();
        nonFermiFarFieldSolve();
        rejectionAndFailure();
        profileRefinement();
        std::cout << "Obstacle-map numerical tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Obstacle-map test failure: " << error.what() << '\n';
        return 1;
    }
}
