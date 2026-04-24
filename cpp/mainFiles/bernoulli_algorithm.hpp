#ifndef BERNOULLI_ALGORITHM_HPP
#define BERNOULLI_ALGORITHM_HPP

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "finiteElement.hpp"
#include "baseProblem.hpp"
#include "levelSet.hpp"
#include "paraview.hpp"
#include "integrationFunFEM.hpp"

namespace bernoulli {

using Mesh = Mesh2;
using FESpace = FESpace2;
using CutSpace = CutFESpaceT2;
using Fun = FunFEM<Mesh>;
using Problem = CutFEM<Mesh>;

using Test = TestFunction<Mesh>;
using ScalarFunction = double (*)(R2, int);

struct Data {
    ScalarFunction rhs = nullptr;               // f in -Delta u = f
    ScalarFunction grad_rhs_x = nullptr;         // x-derivative of f
    ScalarFunction grad_rhs_y = nullptr;         // y-derivative of f
    ScalarFunction fixed_dirichlet = nullptr;   // g_D on fixed boundary
    ScalarFunction fixed_neumann = nullptr;     // g_N on fixed boundary
    ScalarFunction initial_level_set = nullptr; // phi < 0 in the physical domain
};

struct Options {
    int nx = 40;
    int max_iterations = 30;
    int transport_steps = 8;
    int vtk_every = 0;
    double lx = 1.0;
    double ly = 1.0;
    double nitsche_penalty = 50.0;
    double ghost_penalty = 1.0e-1;
    double velocity_boundary_penalty = 50.0;
    double time_step = 0.02;
    double gradient_tolerance = 1.0e-10;
    std::string solver_name = "default";
    std::string output_prefix = "bernoulli";
};

struct IterationInfo {
    int iteration = 0;
    double boundary_residual = 0.0;
    double velocity_norm = 0.0;
    double mean_zero_radius = -1.0;
};

inline ProblemOption make_problem_option(const Options& opt) {
    ProblemOption po;
    po.solver_name_ = opt.solver_name;
    po.verbose_ = 0;
    po.order_space_element_quadrature_ = 5;
    po.order_space_bord_quadrature_ = 5;
    return po;
}

struct CutGeometry {
    std::unique_ptr<InterfaceLevelSet<Mesh>> interface;
    std::unique_ptr<ActiveMesh<Mesh>> domain;
    std::unique_ptr<CutSpace> space;

    // IMPORTANT:
    // FunFEM stores a reference/pointer to the FE space it is built on.
    // The velocity FunFEM returned by solve_velocity() is used later in
    // h1_norm_velocity() and transport_level_set(), so its space must outlive
    // solve_velocity().  Keeping it here avoids a dangling reference.
    std::unique_ptr<CutSpace> velocity_space;
};

class BernoulliAlgorithm {
  public:
    BernoulliAlgorithm(const Data& data, const Options& options)
        : data_(data), opt_(options), Th_(options.nx, options.nx, 0.0, 0.0, options.lx, options.ly),
          Lh_(Th_, DataFE<Mesh>::P1), Vh_(Th_, DataFE<Mesh>::P1), velocity_fe_(1), Bh_(Th_, velocity_fe_),
          level_set_(Lh_, data.initial_level_set) {
        if (!data_.rhs || !data_.grad_rhs_x || !data_.grad_rhs_y || !data_.fixed_dirichlet || !data_.fixed_neumann ||
            !data_.initial_level_set) {
            throw std::invalid_argument("BernoulliAlgorithm: all data callbacks must be provided.");
        }
    }

    std::vector<IterationInfo> run() {
        std::vector<IterationInfo> history;
        history.reserve(opt_.max_iterations + 1);

        std::ofstream csv(opt_.output_prefix + "_history.csv");
        csv << "iteration,boundary_residual,velocity_norm,mean_zero_radius\n";

        for (int it = 0; it <= opt_.max_iterations; ++it) {
            auto geometry = build_geometry();
            auto primal = solve_primal(geometry);
            auto adjoint = solve_adjoint(geometry, *primal);
            const double boundary_residual = compute_boundary_residual(geometry, *primal);
            auto velocity = solve_velocity(geometry, *primal, *adjoint);
            const double vnorm = h1_norm_velocity(geometry, *velocity);
            const double radius = estimate_mean_zero_radius(R2(0.5 * opt_.lx, 0.5 * opt_.ly));

            history.push_back({it, boundary_residual, vnorm, radius});
            csv << it << ',' << boundary_residual << ',' << vnorm << ',' << radius << '\n';
            std::cout << "it=" << it << "  boundary residual=" << boundary_residual << "  |beta|_H1=" << vnorm
                      << "  r0≈" << radius << std::endl;

            if (opt_.vtk_every > 0 && it % opt_.vtk_every == 0) {
                write_vtk(geometry, *primal, it);
                write_levelset_vtk(it);
                write_levelset_contour_dat(it);
            }
            if (it == opt_.max_iterations || vnorm < opt_.gradient_tolerance) break;
            transport_level_set(geometry, *velocity);
        }
        return history;
    }

    const Fun& level_set() const { return level_set_; }
    const Mesh& mesh() const { return Th_; }

  private:
    Data data_;
    Options opt_;
    Mesh Th_;
    FESpace Lh_;
    FESpace Vh_;
    Lagrange2 velocity_fe_;
    FESpace Bh_;
    Fun level_set_;

    CutGeometry build_geometry() const {
        CutGeometry g;
        g.interface = std::make_unique<InterfaceLevelSet<Mesh>>(Th_, level_set_);
        g.domain = std::make_unique<ActiveMesh<Mesh>>(Th_);
        g.domain->truncate(*g.interface, 1); // keep phi < 0; discard the positive side
        g.space = std::make_unique<CutSpace>(*g.domain, Vh_);

        // Velocity is also solved on the active cut mesh so that u_h and p_h,
        // which are cut-space functions, are never evaluated on inactive
        // background cells.  The space is owned by CutGeometry to preserve
        // lifetime for the returned velocity FunFEM.
        g.velocity_space = std::make_unique<CutSpace>(*g.domain, Bh_);
        return g;
    }

    void add_cut_poisson_operator(Problem& problem, const CutGeometry& g) const {
        const MeshParameter& h(Parameter::h);
        Normal n;
        Test u(*g.space, 1), v(*g.space, 1);
        Test dun = grad(u) * n;
        Test dvn = grad(v) * n;

        problem.addBilinear(innerProduct(grad(u), grad(v)), *g.domain);
        problem.addBilinear(-innerProduct(dun, v) - innerProduct(u, dvn) +
                                innerProduct((opt_.nitsche_penalty / h) * u, v),
                            *g.interface);
        problem.addFaceStabilization(
            innerProduct((opt_.ghost_penalty * h) * jump(grad(u) * n), jump(grad(v) * n)), *g.domain);
    }

    std::unique_ptr<Fun> solve_primal(const CutGeometry& g) const {
        Problem primal(*g.space, make_problem_option(opt_));
        add_cut_poisson_operator(primal, g);

        Fun f(*g.space, data_.rhs);
        Fun gN(*g.space, data_.fixed_neumann);
        Test v(*g.space, 1);
        primal.addLinear(innerProduct(f.expr(), v), *g.domain);
        primal.addLinear(innerProduct(gN.expr(), v), *g.domain, INTEGRAL_BOUNDARY);
        primal.solve();
        auto sol = std::make_unique<Fun>(*g.space);
        sol->init(primal.rhs_);
        return sol;
    }

    std::unique_ptr<Fun> solve_adjoint(const CutGeometry& g, const Fun& primal) const {
        Problem adjoint(*g.space, make_problem_option(opt_));
        add_cut_poisson_operator(adjoint, g);

        const MeshParameter& h(Parameter::h);
        Fun gD(*g.space, data_.fixed_dirichlet);
        Test v(*g.space, 1);
        adjoint.addLinear(innerProduct(primal.expr() - gD.expr(), (1.0 / h) * v), *g.domain, INTEGRAL_BOUNDARY);
        adjoint.solve();
        auto sol = std::make_unique<Fun>(*g.space);
        sol->init(adjoint.rhs_);
        return sol;
    }

    std::unique_ptr<Fun> solve_velocity(const CutGeometry& g, const Fun& primal, const Fun& adjoint) const {
        Problem velocity(*g.velocity_space, make_problem_option(opt_));

        Test bx(*g.velocity_space, 1, 0), by(*g.velocity_space, 1, 1);
        Test tx(*g.velocity_space, 1, 0), ty(*g.velocity_space, 1, 1);

        velocity.addBilinear(innerProduct(grad(bx), grad(tx)) + innerProduct(bx, tx), *g.domain);
        velocity.addBilinear(innerProduct(grad(by), grad(ty)) + innerProduct(by, ty), *g.domain);

        Fun f(*g.space, data_.rhs);
        Fun grad_fx(*g.space, data_.grad_rhs_x);
        Fun grad_fy(*g.space, data_.grad_rhs_y);

        auto ux = dx(primal.expr());
        auto uy = dy(primal.expr());
        auto px = dx(adjoint.expr());
        auto py = dy(adjoint.expr());
        auto p = adjoint.expr();
        auto ff = f.expr();
        auto gfx = grad_fx.expr();
        auto gfy = grad_fy.expr();

        auto coeff_div = ff * p - ux * px - uy * py;

        // Assemble -D J(theta), where D J is the continuous domain shape derivative
        // evaluated with the current CutFEM primal/adjoint fields.
        velocity.addLinear(-innerProduct(coeff_div, dx(tx) + dy(ty)), *g.domain);
        velocity.addLinear(-innerProduct(2.0 * ux * px, dx(tx)), *g.domain);
        velocity.addLinear(-innerProduct(ux * py, dy(tx) + dx(ty)), *g.domain);
        velocity.addLinear(-innerProduct(uy * px, dy(tx) + dx(ty)), *g.domain);
        velocity.addLinear(-innerProduct(2.0 * uy * py, dy(ty)), *g.domain);
        velocity.addLinear(-innerProduct(gfx * p, tx) - innerProduct(gfy * p, ty), *g.domain);

        velocity.solve();
        auto sol = std::make_unique<Fun>(*g.velocity_space);
        sol->init(velocity.rhs_);
        return sol;
    }

    double compute_boundary_residual(const CutGeometry& g, const Fun& primal) const {
        // Diagnostic only: assemble the fixed-boundary residual functional against P1 test functions
        // and report the Euclidean norm of the resulting load vector. The optimization step itself
        // uses the adjoint/shape derivative, not this scalar.
        Problem residual(*g.space, make_problem_option(opt_));
        Fun gD(*g.space, data_.fixed_dirichlet);
        Test q(*g.space, 1);
        residual.addLinear(innerProduct(primal.expr() - gD.expr(), q), *g.domain, INTEGRAL_BOUNDARY);
        double sum = 0.0;
        for (int i = 0; i < residual.rhs_.size(); ++i) sum += residual.rhs_(i) * residual.rhs_(i);
        return std::sqrt(sum);
    }

    double h1_norm_velocity(const CutGeometry& g, const Fun& velocity) const {
        auto bx = velocity.expr(0);
        auto by = velocity.expr(1);
        const double val = integral(*g.domain,
                                    bx * bx + by * by + dx(bx) * dx(bx) + dy(bx) * dy(bx) + dx(by) * dx(by) +
                                        dy(by) * dy(by));
        return std::sqrt(std::max(0.0, val));
    }

    void transport_level_set(const CutGeometry& g, const Fun& velocity) {
        // LevelSet::move expects a velocity defined on the full background FE
        // space.  Here beta is intentionally defined on the active cut velocity
        // space, so we update the background level-set nodes explicitly while
        // only evaluating beta on active cut elements.
        const double dt = opt_.time_step / std::max(1, opt_.transport_steps);

        for (int step = 0; step < opt_.transport_steps; ++step) {
            KN<double> delta(level_set_.v.size());
            KN<double> count(level_set_.v.size());
            delta = 0.;
            count = 0.;

            for (int k = g.domain->first_element(); k < g.domain->last_element(); k += g.domain->next_element()) {
                if (g.domain->isInactive(k, 0)) continue;

                const int kb = g.domain->idxElementInBackMesh(k);
                const auto& K = Th_[kb];

                for (int a = 0; a < Mesh::Element::nv; ++a) {
                    const R2 P = K[a];
                    const int inode = Th_(K[a]);
                    if (inode < 0 || inode >= level_set_.v.size()) continue;

                    const double beta_x = velocity.eval(k, P, 0, op_id);
                    const double beta_y = velocity.eval(k, P, 1, op_id);
                    const double phi_x = level_set_.eval(kb, P, 0, op_dx);
                    const double phi_y = level_set_.eval(kb, P, 0, op_dy);

                    delta(inode) += -(beta_x * phi_x + beta_y * phi_y) * dt;
                    count(inode) += 1.;
                }
            }

            for (int i = 0; i < level_set_.v.size(); ++i) {
                if (count(i) > 0.) level_set_.v(i) += delta(i) / count(i);
            }
        }
    }

    double estimate_mean_zero_radius(const R2 center) const {
        double sum = 0.0;
        int count = 0;
        for (int k = Th_.first_element(); k < Th_.last_element(); k += Th_.next_element()) {
            const auto& K = Th_[k];
            for (int e = 0; e < 3; ++e) {
                const R2& A = K[Triangle2::nvedge[e][0]];
                const R2& B = K[Triangle2::nvedge[e][1]];
                const int ia = Th_(K[Triangle2::nvedge[e][0]]);
                const int ib = Th_(K[Triangle2::nvedge[e][1]]);
                if (ia >= level_set_.v.size() || ib >= level_set_.v.size()) continue;
                const double fa = level_set_.v(ia);
                const double fb = level_set_.v(ib);
                if (fa == 0.0 || fa * fb < 0.0) {
                    const double t = std::abs(fa) / (std::abs(fa) + std::abs(fb) + 1e-30);
                    const R2 P = (1.0 - t) * A + t * B;
                    sum += std::sqrt((P.x - center.x) * (P.x - center.x) + (P.y - center.y) * (P.y - center.y));
                    ++count;
                }
            }
        }
        return count > 0 ? sum / count : -1.0;
    }

    void write_vtk(const CutGeometry& g, Fun& primal, int iteration) const {
        const std::string name = opt_.output_prefix + "_" + std::to_string(iteration) + ".vtk";
        Paraview<Mesh> writer(*g.domain, name.c_str());
        writer.add(primal, "u", 0, 1);
    }

    void write_levelset_vtk(int iteration) {
        const std::string name = opt_.output_prefix + "_levelset_" + std::to_string(iteration) + ".vtk";
        Paraview<Mesh> writer(Th_, name.c_str());
        writer.add(level_set_, "phi", 0, 1);
    }

    void write_levelset_contour_dat(int iteration) const {
        const std::string name = opt_.output_prefix + "_shape_" + std::to_string(iteration) + ".dat";
        std::ofstream out(name);
        if (!out) return;

        auto push_unique = [](std::vector<R2>& pts, const R2& P) {
            const double tol = 1e-12;
            for (const auto& Q : pts) {
                const double dx = P.x - Q.x;
                const double dy = P.y - Q.y;
                if (dx * dx + dy * dy < tol * tol) return;
            }
            pts.push_back(P);
        };

        for (int k = Th_.first_element(); k < Th_.last_element(); k += Th_.next_element()) {
            const auto& K = Th_[k];
            std::vector<R2> pts;

            for (int e = 0; e < 3; ++e) {
                const int ia_loc = Triangle2::nvedge[e][0];
                const int ib_loc = Triangle2::nvedge[e][1];

                const R2& A = K[ia_loc];
                const R2& B = K[ib_loc];

                const int ia = Th_(K[ia_loc]);
                const int ib = Th_(K[ib_loc]);

                if (ia < 0 || ib < 0 || ia >= level_set_.v.size() || ib >= level_set_.v.size()) continue;

                const double fa = level_set_.v(ia);
                const double fb = level_set_.v(ib);

                if (std::abs(fa) < 1e-14 && std::abs(fb) < 1e-14) {
                    out << A.x << " " << A.y << "\\n";
                    out << B.x << " " << B.y << "\\n\\n";
                    continue;
                }

                if (std::abs(fa) < 1e-14) {
                    push_unique(pts, A);
                } else if (std::abs(fb) < 1e-14) {
                    push_unique(pts, B);
                } else if (fa * fb < 0.0) {
                    const double t = fa / (fa - fb);
                    const R2 P = (1.0 - t) * A + t * B;
                    push_unique(pts, P);
                }
            }

            if (pts.size() == 2) {
                out << pts[0].x << " " << pts[0].y << "\\n";
                out << pts[1].x << " " << pts[1].y << "\\n\\n";
            }
        }
    }
};

} // namespace bernoulli

#endif // BERNOULLI_ALGORITHM_HPP
