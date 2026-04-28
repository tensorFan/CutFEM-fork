// DOESNT WORK ATM
// -----------------------------------------------------------------------------
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "bernoulli_algorithm.hpp"
#ifdef USE_MPI
#include "cfmpi.hpp"
#endif

namespace {

constexpr double cx = 0.5;
constexpr double cy = 0.5;
constexpr double initial_radius = 0.125;
constexpr double exact_radius = 0.25;

inline double radius(const R2& P) {
    const double dx = P.x - cx;
    const double dy = P.y - cy;
    return std::sqrt(dx * dx + dy * dy);
}

double rhs(const R2 P, int) {
    return -4.0 / std::max(radius(P), 1.0e-12);
}

double grad_rhs_x(const R2 P, int) {
    const double dx = P.x - cx;
    const double r = std::max(radius(P), 1.0e-12);
    return 4.0 * dx / (r * r * r);
}

double grad_rhs_y(const R2 P, int) {
    const double dy = P.y - cy;
    const double r = std::max(radius(P), 1.0e-12);
    return 4.0 * dy / (r * r * r);
}

double fixed_dirichlet(const R2 P, int) {
    return 4.0 * radius(P) - 1.0;
}

double fixed_neumann(const R2 P, int) {
    const double r = std::max(radius(P), 1.0e-12);
    const double ux = 4.0 * (P.x - cx) / r;
    const double uy = 4.0 * (P.y - cy) / r;

    // Outward unit normal to the unit-square fixed boundary.
    if (std::abs(P.x) < 1e-12) return -ux;
    if (std::abs(P.x - 1.0) < 1e-12) return ux;
    if (std::abs(P.y) < 1e-12) return -uy;
    if (std::abs(P.y - 1.0) < 1e-12) return uy;
    return 0.0;
}

double initial_level_set_fun(const R2 P, int) {
    return initial_radius - radius(P); // positive inside the obstacle, negative in the physical domain
}

} // namespace

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPIcf mpi(argc, argv);
#else
    (void)argc;
    (void)argv;
#endif

    bernoulli::Data data;
    data.rhs = rhs;
    data.grad_rhs_x = grad_rhs_x;
    data.grad_rhs_y = grad_rhs_y;
    data.fixed_dirichlet = fixed_dirichlet;
    data.fixed_neumann = fixed_neumann;
    data.initial_level_set = initial_level_set_fun;

    bernoulli::Options opt;
    opt.nx = 40;
    opt.max_iterations = 30;
    opt.transport_steps = 8;
    opt.damping_alpha = 0.5;
    opt.transport_steps = 3;
    opt.output_prefix = "bernoulli_circle";
    opt.vtk_every = 1;

    bernoulli::BernoulliAlgorithm solver(data, opt);
    const auto history = solver.run();

    if (!history.empty()) {
        const auto& last = history.back();
        std::cout << "\nTarget radius: " << exact_radius << "\n"
                  << "Final estimated radius: " << last.mean_zero_radius << "\n"
                  << "History written to: " << opt.output_prefix << "_history.csv\n";
    }
    return EXIT_SUCCESS;
}
