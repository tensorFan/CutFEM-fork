#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "cutFEMConfig.h"
#ifdef USE_MPI
#include "cfmpi.hpp"
#endif
#include "finiteElement.hpp"
#include "baseProblem.hpp"
#include "paraview.hpp"
#include "../common/SparseMatMap.hpp"
#include "umfpack.h"

// Compute one real Maxwell eigenpair A x = lambda B x, x=(u,p[,eta]),
// using an unfitted Ned0-P1 Kikuchi formulation with mixed ghost stabilization.
// The pressure and optional harmonic multiplier have zero mass in B.
// Shifted inverse iteration solves (A-target*B)y=B*x without inverting B.
// Requires USE_UMFPACK=ON; run on one MPI rank (or in a serial build).
//
// From the repository root:
//   cmake --build build --target maxwell3D_kikuchi_eigenfunction -j4
//   ./build/bin/maxwell3D_kikuchi_eigenfunction --output build/maxwell_mode.vtk
//   ./build/bin/maxwell3D_kikuchi_eigenfunction --example spherical_shell \
//       --nx 19 --target 3.2 --output build/shell_mode.vtk
//
// Open the companion *_surface.vtk in ParaView for Surface/Clip views.
// Use the volume VTK for Slice/Glyph filters with the eigenfunction vector.
// Colour by eigenfunction (Magnitude). The exported field
// has unit physical L2 norm; pressure is scaled by the same factor.

namespace {
using Mesh = Mesh3;
using Space = FESpace3;
using CutSpace = CutFESpaceT3;
using Fun_h = FunFEM<Mesh>;
using FunTest = TestFunction<Mesh>;

struct Config {
    bool shell = true;
    int nx = 19;
    double target = 3.2;
    double penalty = 10.;
    double tolerance = 1e-8;
    int max_iterations = 500;
    std::string output = "maxwell_kikuchi_eigenfunction.vtk";
};

void usage(const char *exe) {
    std::cout << "Usage: " << exe << " [options]\n"
              << "  --example cube|spherical_shell   default spherical_shell\n"
              << "  --nx N            background vertices per direction, default 19\n"
              << "  --target X        eigenvalue shift, default 2 (cube), 3.2 (shell)\n"
              << "  --penalty X       Nitsche penalty coefficient, default 10\n"
              << "  --tol X           relative eigenpair residual, default 1e-8\n"
              << "  --max-it N        inverse iteration limit, default 500\n"
              << "  --output FILE.vtk output on the physical cut mesh\n"
              << "  --help\n"
              << "Computes one mode near the shift; lambda is frequency squared\n"
              << "for eps=mu=1. Requires UMFPACK and a single MPI rank.\n";
}

bool parse_args(int argc, char **argv, Config &cfg) {
    bool target_given = false;
    for (int a = 1; a < argc; ++a) {
        const std::string key(argv[a]);
        if (key == "--help" || key == "-h") {
            usage(argv[0]);
            return false;
        }
        auto value = [&]() {
            if (++a >= argc) throw std::runtime_error("Missing value after " + key);
            return std::string(argv[a]);
        };
        auto integer = [&]() {
            const auto s = value();
            std::size_t end;
            const int n = std::stoi(s, &end);
            if (end != s.size()) throw std::runtime_error("Invalid integer for " + key);
            return n;
        };
        auto real = [&]() {
            const auto s = value();
            std::size_t end;
            const double x = std::stod(s, &end);
            if (end != s.size() || !std::isfinite(x))
                throw std::runtime_error("Invalid finite number for " + key);
            return x;
        };
        if (key == "--example") {
            const auto ex = value();
            if (ex != "cube" && ex != "spherical_shell")
                throw std::runtime_error("Unknown example: " + ex);
            cfg.shell = (ex == "spherical_shell");
        } else if (key == "--nx") {
            cfg.nx = integer();
        } else if (key == "--target") {
            cfg.target = real();
            target_given = true;
        } else if (key == "--tol") {
            cfg.tolerance = real();
        } else if (key == "--penalty") {
            cfg.penalty = real();
        } else if (key == "--max-it") {
            cfg.max_iterations = integer();
        } else if (key == "--output") {
            cfg.output = value();
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }
    if (!target_given) cfg.target = cfg.shell ? 3.2 : 2.;
    if (cfg.nx < 4 || (cfg.shell && cfg.nx < 7))
        throw std::runtime_error("Use --nx >= 4 for the cube or >= 7 for the shell.");
    if (cfg.target <= 0. || cfg.penalty <= 0. || cfg.tolerance <= 0. || cfg.tolerance >= 1. || cfg.max_iterations < 1)
        throw std::runtime_error("Require target and penalty > 0, 0 < tol < 1, and max-it >= 1.");
    if (!cfg.output.ends_with(".vtk"))
        throw std::runtime_error("The output filename must end in .vtk.");
    return true;
}

R cube_level_set(double *P, int, int) { return M_PI - P[2]; }

R shell_level_set(double *P, int, int) {
    const double x = P[0] - M_PI / 2., y = P[1] - M_PI / 2., z = P[2] - M_PI / 2.;
    const double r = std::sqrt(x*x + y*y + z*z);
    // Positive signed distance to the shell boundary. Interpolating the
    // quartic product of the two sphere equations distorts the cut geometry.
    return std::min(r - M_PI / 5., M_PI / 3. - r);
}

R harmonic_field(double *P, int i, int) {
    const double x = P[0] - M_PI / 2., y = P[1] - M_PI / 2., z = P[2] - M_PI / 2.;
    const double r2 = x*x + y*y + z*z;
    return (P[i] - M_PI / 2.) / (r2 * std::sqrt(r2));
}

R zero(double *, int, int) { return 0.; }

// Reuse a single sparse factorization throughout the inverse iteration.
// SparseMatrixRC is CSR: UMFPACK sees its transpose as CSC, hence UMFPACK_At.
class ShiftSolver {
    SparseMatrixRC<double> matrix_;
    void *numeric_ = nullptr;

public:
    ShiftSolver(int n, const Matrix &matrix) : matrix_(n, n, matrix) {
        void *symbolic = nullptr;
        int status = umfpack_di_symbolic(n, n, matrix_.p, matrix_.j, matrix_.a,
                                         &symbolic, nullptr, nullptr);
        if (status == UMFPACK_OK)
            status = umfpack_di_numeric(matrix_.p, matrix_.j, matrix_.a,
                                        symbolic, &numeric_, nullptr, nullptr);
        umfpack_di_free_symbolic(&symbolic);
        if (status != UMFPACK_OK) {
            umfpack_di_free_numeric(&numeric_);
            throw std::runtime_error("UMFPACK factorization failed (status " + std::to_string(status)
                                     + "). Try a different --target.");
        }
    }
    ~ShiftSolver() { umfpack_di_free_numeric(&numeric_); }
    ShiftSolver(const ShiftSolver &) = delete;
    ShiftSolver &operator=(const ShiftSolver &) = delete;

    void solve(const Rn &rhs, Rn &x) const {
        const int status = umfpack_di_solve(UMFPACK_At, matrix_.p, matrix_.j, matrix_.a,
                                            x, rhs, numeric_, nullptr, nullptr);
        if (status != UMFPACK_OK)
            throw std::runtime_error("UMFPACK solve failed (status " + std::to_string(status) + ").");
    }
};

double dot(const Rn &x, const Rn &y) {
    double result = 0.;
    for (int i = 0; i < x.size(); ++i) result += x[i] * y[i];
    return result;
}

struct EigenpairInfo {
    double lambda;
    double residual;
    int iterations;
};

// Use the planar boundary normals actually used in assembly. For a linear
// trace on a triangle, integral |v|^2 = area/12*(sum |v_i|^2 + |sum v_i|^2).
void report_shell_diagnostics(const InterfaceLevelSet<Mesh> &surface,
                              Fun_h &u, Fun_h &p, const Matrix &A, const Matrix &B,
                              const Rn &coefficients) {
    double tangential2 = 0., trace2 = 0.;
    for (int k = 0; k < surface.nbElement(); ++k) {
        const auto n = surface.normal(k);
        const int ku = u.idxElementFromBackMesh(surface.idxElementOfFace(k), 0);
        R3 sum_t(0., 0., 0.), sum_u(0., 0., 0.);
        double t2 = 0., u2 = 0.;
        for (int j = 0; j < 3; ++j) {
            const auto point = surface(k, j);
            const R3 field(u.eval(ku, point, 0), u.eval(ku, point, 1), u.eval(ku, point, 2));
            const double un = field[0]*n[0] + field[1]*n[1] + field[2]*n[2];
            const R3 tangent = field - un*n;
            for (int d = 0; d < 3; ++d) {
                t2 += tangent[d]*tangent[d];
                u2 += field[d]*field[d];
                sum_t[d] += tangent[d];
                sum_u[d] += field[d];
            }
        }
        for (int d = 0; d < 3; ++d) {
            t2 += sum_t[d]*sum_t[d];
            u2 += sum_u[d]*sum_u[d];
        }
        tangential2 += surface.measure(k)*t2/12.;
        trace2 += surface.measure(k)*u2/12.;
    }
    const int n = coefficients.size();
    Rn bx(n);
    multiply(n, n, B, coefficients, bx);
    double harmonic_residual = 0.;
    for (const auto &[ij, value] : A)
        if (ij.first == n - 1) harmonic_residual += value*coefficients[ij.second];
    std::cout << "Physical diagnostics (unit volume L2 norm):\n"
              << "  boundary tangential L2 = " << std::sqrt(tangential2) << '\n'
              << "  boundary total L2 = " << std::sqrt(trace2) << '\n'
              << "  pressure L2 = " << L2normCut(p, zero, 0, 1) << '\n'
              << "  stabilized mass = " << dot(coefficients, bx) << " (physical mass = 1)\n"
              << "  harmonic constraint residual = " << std::abs(harmonic_residual) << '\n';
}

// Export only the physical boundary: the volume writer duplicates vertices
// per tetrahedron, so ParaView's Surface view also displays internal faces.
// Keep element-local samples here to preserve the Nedelec normal jumps.
void write_shell_surface(const std::string &filename, const InterfaceLevelSet<Mesh> &surface,
                         Fun_h &u, Fun_h &p, const EigenpairInfo &info) {
    std::ofstream out(filename);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    const int nf = surface.nbElement();
    out << std::setprecision(17)
        << "# vtk DataFile Version 3.0\nMaxwell shell boundary\nASCII\nDATASET POLYDATA\n"
        << "POINTS " << 3 * nf << " double\n";
    for (int k = 0; k < nf; ++k)
        for (int j = 0; j < 3; ++j) out << surface(k, j) << '\n';
    out << "POLYGONS " << nf << ' ' << 4 * nf << '\n';
    for (int k = 0; k < nf; ++k) {
        // Orient both boundary components away from the retained region.
        const auto a = surface(k, 1) - surface(k, 0);
        const auto b = surface(k, 2) - surface(k, 0);
        const auto n = surface.normal(k); // Positive level-set gradient points inward.
        const double alignment = (a[1]*b[2]-a[2]*b[1])*n[0]
                               + (a[2]*b[0]-a[0]*b[2])*n[1]
                               + (a[0]*b[1]-a[1]*b[0])*n[2];
        out << "3 " << 3*k << ' ' << 3*k + (alignment > 0. ? 2 : 1)
            << ' ' << 3*k + (alignment > 0. ? 1 : 2) << '\n';
    }
    out << "FIELD FieldData 2\neigenvalue 1 1 double\n" << info.lambda
        << "\nrelative_residual 1 1 double\n" << info.residual
        << "\nPOINT_DATA " << 3 * nf << "\nVECTORS eigenfunction double\n";
    for (int k = 0; k < nf; ++k) {
        const int ku = u.idxElementFromBackMesh(surface.idxElementOfFace(k), 0);
        for (int j = 0; j < 3; ++j) {
            const auto point = surface(k, j);
            out << u.eval(ku, point, 0) << ' ' << u.eval(ku, point, 1)
                << ' ' << u.eval(ku, point, 2) << '\n';
        }
    }
    out << "SCALARS pressure double\nLOOKUP_TABLE default\n";
    for (int k = 0; k < nf; ++k) {
        const int kp = p.idxElementFromBackMesh(surface.idxElementOfFace(k), 0);
        for (int j = 0; j < 3; ++j) out << p.eval(kp, surface(k, j), 0) << '\n';
    }
    out.close();
}

EigenpairInfo inverse_iteration(const Matrix &A, const Matrix &B, int n_u,
                                const Config &cfg, Rn &x) {
    const int n = x.size();
    Matrix shifted(A);
    for (const auto &[ij, value] : B) shifted[ij] -= cfg.target * value;
    const ShiftSolver solve_shifted(n, shifted);
    const SparseMatrixRC<double> a(n, n, A), b(n, n, B);
    auto multiply = [](const SparseMatrixRC<double> &m, const Rn &v, Rn &mv) {
        mv = 0.;
        m.addMatMul(v, mv);
    };
    // Deterministic broad initial data; only u contributes to B*x.
    std::mt19937 generator(1729);
    std::uniform_real_distribution<double> uniform(-1., 1.);
    x = 0.;
    for (int i = 0; i < n_u; ++i) x[i] = uniform(generator);
    Rn bx(n), ax(n), y(n);
    multiply(b, x, bx);
    for (int iteration = 1; iteration <= cfg.max_iterations; ++iteration) {
        solve_shifted.solve(bx, y);
        multiply(b, y, bx);
        const double mass = dot(y, bx);
        if (!std::isfinite(mass) || mass <= 0.)
            throw std::runtime_error("Inverse iteration produced a nonpositive or nonfinite mass norm.");
        const double scale = 1. / std::sqrt(mass);
        for (int i = 0; i < n; ++i) {
            x[i] = y[i] * scale;
            bx[i] *= scale;
        }
        multiply(a, x, ax);
        const double lambda = dot(x, ax) / dot(x, bx);
        double residual2 = 0.;
        for (int i = 0; i < n; ++i) {
            const double r = ax[i] - lambda * bx[i];
            residual2 += r*r;
        }
        // Include pressure and harmonic rows in the convergence check.
        const double denominator = std::sqrt(dot(ax, ax)) + std::abs(lambda) * std::sqrt(dot(bx, bx));
        const double residual = denominator > 0. ? std::sqrt(residual2) / denominator : 1.;
        if (!std::isfinite(lambda) || !std::isfinite(residual))
            throw std::runtime_error("Nonfinite eigenpair or residual during inverse iteration.");
        if (iteration == 1 || iteration % 10 == 0 || residual < cfg.tolerance)
            std::cout << "Iteration " << iteration << ": lambda=" << std::setprecision(12) << lambda
                      << ", relative residual=" << residual << std::endl;
        if (residual < cfg.tolerance) {
            if (lambda <= 0.)
                throw std::runtime_error("The converged mode is not positive; choose another --target.");
            return {lambda, residual, iteration};
        }
    }
    throw std::runtime_error("Eigenpair did not converge; adjust --target or increase --max-it. No VTK written.");
}

void compute(const Config &cfg) {
    const double h = M_PI / (cfg.nx - 1);
    // Cut the cube's top face through the final layer, away from grid vertices.
    // The other five cube faces are fitted. The shell has two unfitted faces.
    const double zmax = cfg.shell ? M_PI : M_PI + 0.25 * h;
    // Offset the shell's background grid, keeping the physical shell fixed.
    // On the unshifted grid, r=pi/3 passes through vertices for nx=7,13,...;
    // roundoff there creates collapsed cut tetrahedra in the VTK geometry.
    const double x0 = cfg.shell ? -0.137 * h : 0.;
    const double y0 = cfg.shell ? -0.173 * h : 0.;
    const double z0 = cfg.shell ? -0.193 * h : 0.;
    Mesh Kh(cfg.nx, cfg.nx, cfg.nx, x0, y0, z0, M_PI, M_PI, zmax);
    Space Uh_background(Kh, DataFE<Mesh>::Ned0);
    Space Ph_background(Kh, DataFE<Mesh>::P1);
    Fun_h level_set(Ph_background, cfg.shell ? shell_level_set : cube_level_set);
    InterfaceLevelSet<Mesh> interface(Kh, level_set);
    ActiveMesh<Mesh> Khi(Kh);
    Khi.truncate(interface, -1); // Retain the positive level-set region.
    Khi.info();

    CutSpace Uh(Khi, Uh_background), Ph(Khi, Ph_background);
    CutFEM<Mesh> A(Uh); A.add(Ph);
    CutFEM<Mesh> B(Uh); B.add(Ph);
    FunTest u(Uh, 3, 0), v(Uh, 3, 0), p(Ph, 1, 0), q(Ph, 1, 0);
    Normal normal;
    const double penalty = cfg.penalty, tau_curl = 1., tau_mass = 1., tau_p = 1.;

    // Symmetric Kikuchi/Nitsche form, eps=mu=1, with zero pressure mass.
    // The mixed ghost terms extend the gradient constraint to the active mesh.
    // Do not add a grad(p)-grad(q) patch term: it relaxes that constraint and
    // introduced nonphysical modes near the first shell eigenvalue in this case.
    A.addBilinear(
        +innerProduct(curl(u), curl(v))
        +innerProduct(grad(p), v)
        +innerProduct(u, grad(q))
    , Khi);
    const auto boundary =
        -innerProduct(u * normal, q)
        -innerProduct(p, v * normal)
        -innerProduct(p, penalty / h * q)
        -innerProduct(curl(u), cross(normal, v))
        -innerProduct(cross(normal, u), curl(v))
        +innerProduct(cross(normal, u), penalty / h * cross(normal, v));
    A.addBilinear(boundary, interface);
    if (!cfg.shell) A.addBilinear(boundary, Khi, INTEGRAL_BOUNDARY);
    A.addPatchStabilization(
        +innerProduct(tau_curl * jump(curl(u)), jump(curl(v)))
        +innerProduct(tau_p * jump(grad(p)), jump(v))
        +innerProduct(tau_p * jump(u), jump(grad(q)))
    , Khi);
    B.addBilinear(+innerProduct(u, v), Khi);
    B.addPatchStabilization(+innerProduct(tau_mass * jump(u), jump(v)), Khi);

    const int n_u = Uh.get_nb_dof(), n_p = Ph.get_nb_dof();
    if (cfg.shell) {
        // Use exactly the mass matrix's patch metric for the harmonic constraint.
        Fun_h harmonic(Uh, harmonic_field);
        Rn harmonic_coefficients(n_u + n_p), row(n_u + n_p);
        harmonic_coefficients = 0.;
        for (int i = 0; i < n_u; ++i) harmonic_coefficients[i] = harmonic.v[i];
        multiply(n_u + n_p, n_u + n_p, B.mat_[0], harmonic_coefficients, row);
        A.addLagrangeVecToRowAndCol(row, row, 0.);
        A.mat_[0][{n_u + n_p, n_u + n_p}] = 0.;
        B.addLagrangeMultiplier(+innerProduct(harmonic.exprList(), 0 * v), 0., Khi);
        B.mat_[0][{n_u + n_p, n_u + n_p}] = 0.;
    }

    std::cout << "Unfitted Kikuchi " << (cfg.shell ? "spherical shell" : "cube")
              << ": " << n_u << " field DoFs, " << n_p << " pressure DoFs, target=" << cfg.target
              << std::endl;
    Rn coefficients(n_u + n_p + (cfg.shell ? 1 : 0));
    const auto info = inverse_iteration(A.mat_[0], B.mat_[0], n_u, cfg, coefficients);

    Rn_ u_data = coefficients(SubArray(n_u, 0));
    Rn_ p_data = coefficients(SubArray(n_p, n_u));
    Fun_h eigenfunction(Uh, u_data), pressure(Ph, p_data);
    const double l2 = L2normCut(eigenfunction, zero, 0, 3);
    if (!std::isfinite(l2) || l2 <= 0.) throw std::runtime_error("Invalid physical field norm.");
    int pivot = 0;
    for (int i = 1; i < n_u; ++i)
        if (std::abs(coefficients[i]) > std::abs(coefficients[pivot])) pivot = i;
    const double scale = (coefficients[pivot] < 0. ? -1. : 1.) / l2;
    coefficients *= scale;
    if (cfg.shell)
        report_shell_diagnostics(interface, eigenfunction, pressure, A.mat_[0], B.mat_[0], coefficients);

    // The writer evaluates each element separately at physical cut vertices,
    // preserving the normal jumps of the Nedelec field instead of averaging.
    std::ofstream check(cfg.output);
    if (!check) throw std::runtime_error("Cannot open output: " + cfg.output);
    check.close();
    Paraview<Mesh> writer(Khi, cfg.output, 17);
    {
        std::ofstream metadata(cfg.output, std::ios::app);
        metadata << std::setprecision(17)
                 << "FIELD FieldData 2\n"
                 << "eigenvalue 1 1 double\n" << info.lambda << '\n'
                 << "relative_residual 1 1 double\n" << info.residual << '\n';
        if (!metadata) throw std::runtime_error("Cannot write VTK metadata: " + cfg.output);
    }
    writer.add(eigenfunction, "eigenfunction", 0, 3);
    writer.add(pressure, "pressure", 0, 1);
    if (cfg.shell) {
        const std::string surface_file = cfg.output.substr(0, cfg.output.size() - 4) + "_surface.vtk";
        write_shell_surface(surface_file, interface, eigenfunction, pressure, info);
        std::cout << "Exported boundary for Surface/Clip views: " << surface_file << '\n';
    }
    std::cout << "Exported " << cfg.output << " (unit L2 field norm)\n"
              << "lambda=" << std::setprecision(12) << info.lambda
              << ", sqrt(lambda)=" << std::sqrt(info.lambda)
              << ", relative residual=" << info.residual
              << ", iterations=" << info.iterations << std::endl;
}
} // namespace

int main(int argc, char **argv) {
    try {
        Config cfg;
        if (!parse_args(argc, argv, cfg)) return 0;
#ifdef USE_MPI
        MPIcf mpi(argc, argv);
        if (MPIcf::size() != 1) {
            if (MPIcf::IamMaster())
                std::cerr << "This example requires one MPI rank; run directly or with mpiexec -n 1.\n";
            return 1;
        }
#endif
        compute(cfg);
    } catch (const std::exception &e) {
        std::cerr << "maxwell3D_kikuchi_eigenfunction: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
