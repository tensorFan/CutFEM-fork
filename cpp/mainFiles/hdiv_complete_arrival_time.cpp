#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef USE_MPI
#include "cfmpi.hpp"
#endif

#include "finiteElement.hpp"
#include "baseProblem.hpp"
#include "paraview.hpp"
#include "../num/DA.hpp"

/*
  H(div)-conforming test driver for the convexified complete-arrival-time equation.

  Convex variable:

      w = exp(-u/eps^2).

  Regularized arrival-time equation in w:

      -div( Dw / sqrt(|Dw|^2 + w^2/eps^2) )
      +       w /(eps^2 sqrt(|Dw|^2 + w^2/eps^2)) = f.

  Define

      a(w) = 1 / sqrt(|Dw|^2 + w^2/eps^2),
      p    = -a Dw,
      m    =  a w/eps^2.

  Then

      a^{-1} p + Dw = 0,
      div p + m = f,
      |p|^2 + eps^2 m^2 = 1       when a is evaluated from the same exact w.

  The finite element method below uses p_h in RT0 and w_h in P0, so the
  conservative balance div p_h + m_h = f_h is the primary diagnostic.

  Examples:
    0. 2D exact slab translator / elliptic-regularization toy, f=0.
    1. 2D genuinely wavy manufactured solution, f computed analytically.
    2. 3D true MCF grim-reaper-cylinder translator, direct arrival-time form.
       This is the visually useful one: contour u_h in ParaView.  It avoids
       the exponential dynamic range of the finite-eps w-variable.

  Add to cpp/CMakeLists.txt inside if(${CUTFEM_BUILD_MAIN}):

      set(EX_NAME hdiv_complete_arrival_time)
      add_executable(${EX_NAME} mainFiles/hdiv_complete_arrival_time.cpp)
      target_link_libraries(${EX_NAME} PUBLIC ${EXTRA_LIBS} ${MPI_CXX_LIBRARIES})

  Run:

      ./bin/hdiv_complete_arrival_time        # all examples
      ./bin/hdiv_complete_arrival_time 2      # only the 3D visual example
*/

// -----------------------------------------------------------------------------
// 2D examples
// -----------------------------------------------------------------------------

using Mesh2T    = Mesh2;
using Space2    = FESpace2;
using FunTest2  = TestFunction<Mesh2T>;
using FunFEM2   = FunFEM<Mesh2T>;

namespace HdivArrival2D {

static int example_id = 0;
static double eps = 0.20;
static double xmin = -0.30, xmax = 0.30, ymin = 0.0, ymax = 1.0;
static constexpr double pi = 3.141592653589793238462643383279502884;

struct Bounds {
    double x0, x1, y0, y1;
};

std::string example_name() {
    if (example_id == 0) return "exact_slab_translator_f0";
    if (example_id == 1) return "wavy_2d_manufactured";
    return "unknown_2d";
}

Bounds set_example(int id) {
    example_id = id;
    if (id == 0) {
        eps = 0.25;
        const double R = 0.30;
        xmin = -R; xmax = R; ymin = 0.0; ymax = 1.0;
    } else {
        eps = 0.18;
        xmin = 0.0; xmax = 1.0; ymin = 0.0; ymax = 1.0;
    }
    return {xmin, xmax, ymin, ymax};
}

inline void exact_w_derivatives(const R2 P,
                                double &w, double &wx, double &wy,
                                double &wxx, double &wxy, double &wyy) {
    const double x = P.x;
    const double y = P.y;

    if (example_id == 0) {
        const double R = 0.5 * (xmax - xmin);
        const double cR = std::cos(R / eps);
        const double s  = x / eps;
        const double c  = std::cos(s);
        const double sn = std::sin(s);

        w   = cR / c;
        wx  = (w / eps) * (sn / c);
        wy  = 0.0;
        wxx = (w / (eps * eps)) * (1.0 + 2.0 * (sn / c) * (sn / c));
        wxy = 0.0;
        wyy = 0.0;
        return;
    }

    const double A  = 0.25;
    const double k  = 2.0 * pi;
    const double cx = std::cos(k * x);
    const double sx = std::sin(k * x);
    const double cy = std::cos(k * y);
    const double sy = std::sin(k * y);

    w   = 1.0 + A * cx * cy;
    wx  = -A * k * sx * cy;
    wy  = -A * k * cx * sy;
    wxx = -A * k * k * cx * cy;
    wyy = -A * k * k * cx * cy;
    wxy =  A * k * k * sx * sy;
}

inline double coeff_a(const R2 P) {
    double w, wx, wy, wxx, wxy, wyy;
    exact_w_derivatives(P, w, wx, wy, wxx, wxy, wyy);
    return 1.0 / std::sqrt(wx * wx + wy * wy + (w * w) / (eps * eps));
}

R fun_exact_w(const R2 P, int comp, int dom) {
    double w, wx, wy, wxx, wxy, wyy;
    exact_w_derivatives(P, w, wx, wy, wxx, wxy, wyy);
    return w;
}

R fun_exact_u(const R2 P, int comp, int dom) {
    const double w = fun_exact_w(P, comp, dom);
    return -eps * eps * std::log(w);
}

R fun_exact_flux(const R2 P, int comp, int dom) {
    double w, wx, wy, wxx, wxy, wyy;
    exact_w_derivatives(P, w, wx, wy, wxx, wxy, wyy);
    const double a = coeff_a(P);
    if (comp == 0) return -a * wx;
    return -a * wy;
}

R fun_exact_m(const R2 P, int comp, int dom) {
    double w, wx, wy, wxx, wxy, wyy;
    exact_w_derivatives(P, w, wx, wy, wxx, wxy, wyy);
    const double a = coeff_a(P);
    return a * w / (eps * eps);
}

R fun_ainv(const R2 P, int comp, int dom) { return 1.0 / coeff_a(P); }
R fun_reaction(const R2 P, int comp, int dom) { return coeff_a(P) / (eps * eps); }

R fun_div_exact_flux(const R2 P, int comp, int dom) {
    double w, wx, wy, wxx, wxy, wyy;
    exact_w_derivatives(P, w, wx, wy, wxx, wxy, wyy);

    const double a = coeff_a(P);
    const double Bx = 2.0 * (wx * wxx + wy * wxy) + 2.0 * w * wx / (eps * eps);
    const double By = 2.0 * (wx * wxy + wy * wyy) + 2.0 * w * wy / (eps * eps);
    const double ax = -0.5 * a * a * a * Bx;
    const double ay = -0.5 * a * a * a * By;

    return -(ax * wx + ay * wy) - a * (wxx + wyy);
}

R fun_source(const R2 P, int comp, int dom) {
    return fun_div_exact_flux(P, comp, dom) + fun_exact_m(P, comp, dom);
}

inline R2 to_R2(double *P) { return R2(P[0], P[1]); }

R fun_exact_w_ptr(double *P, int comp) { return fun_exact_w(to_R2(P), comp, 0); }
R fun_exact_u_ptr(double *P, int comp) { return fun_exact_u(to_R2(P), comp, 0); }
R fun_exact_flux_ptr(double *P, int comp) { return fun_exact_flux(to_R2(P), comp, 0); }
R fun_exact_m_ptr(double *P, int comp) { return fun_exact_m(to_R2(P), comp, 0); }
R fun_ainv_ptr(double *P, int comp) { return fun_ainv(to_R2(P), comp, 0); }
R fun_reaction_ptr(double *P, int comp) { return fun_reaction(to_R2(P), comp, 0); }
R fun_source_ptr(double *P, int comp) { return fun_source(to_R2(P), comp, 0); }
R fun_one_ptr(double *P, int comp) { return 1.0; }
R fun_zero_ptr(double *P, int comp) { return 0.0; }

} // namespace HdivArrival2D

struct RunResult2D {
    double h;
    double err_p;
    double err_w;
    double err_u;
    double err_m;
    double err_balance;
    double max_balance;
    double max_cone;
    double mass;
};

RunResult2D run_one_mesh_2d(int nx, int ny, int level, bool write_vtk) {
    using namespace HdivArrival2D;

    Mesh2T Kh(nx, ny, xmin, ymin, xmax, ymax);
    const double h = (xmax - xmin) / static_cast<double>(nx - 1);

    Space2 Vh(Kh, DataFE<Mesh2T>::RT0);
    Space2 Qh(Kh, DataFE<Mesh2T>::P0);

    Lagrange2 FEvecExact(4);
    Space2 Vex(Kh, FEvecExact);
    Space2 Qex(Kh, DataFE<Mesh2T>::P2);

    Normal n;

    FEM<Mesh2T> prob(Vh);
    prob.add(Qh);

    FunFEM2 aInv(Qex, fun_ainv_ptr);
    FunFEM2 reaction(Qex, fun_reaction_ptr);
    FunFEM2 source(Qex, fun_source_ptr);
    FunFEM2 wD(Qex, fun_exact_w_ptr);
    FunFEM2 one(Qex, fun_one_ptr);

    FunTest2 p(Vh, 2), v(Vh, 2);
    FunTest2 w(Qh, 1), q(Qh, 1);

    prob.addBilinear(
        +innerProduct(aInv.expr() * p, v)
        -innerProduct(w, div(v))
        +innerProduct(div(p), q)
        +innerProduct(reaction.expr() * w, q),
        Kh);

    prob.addLinear(+innerProduct(source.expr(), q), Kh);
    prob.addLinear(-innerProduct(wD.expr(), v * n), Kh, INTEGRAL_BOUNDARY);

    prob.solve("umfpack");

    const int ndof_p = Vh.get_nb_dof();
    const int ndof_w = Qh.get_nb_dof();
    Rn_ data_p = prob.rhs_(SubArray(ndof_p, 0));
    Rn_ data_w = prob.rhs_(SubArray(ndof_w, ndof_p));

    Rn data_u(ndof_w, 0.0);
    for (int i = 0; i < ndof_w; ++i) {
        const double wi = (data_w[i] > 1e-300) ? data_w[i] : 1e-300;
        data_u[i] = -eps * eps * std::log(wi);
    }

    FunFEM2 ph(Vh, data_p);
    FunFEM2 wh(Qh, data_w);
    FunFEM2 uh(Qh, data_u);

    auto div_ph = dx(ph.expr(0)) + dy(ph.expr(1));
    auto mh     = reaction.expr() * wh.expr();
    auto balance = div_ph + mh - source.expr();
    auto cone    = ph.expr(0) * ph.expr(0) + ph.expr(1) * ph.expr(1)
                 + (eps * eps) * mh * mh - one.expr();

    const double err_p = L2norm(ph, fun_exact_flux_ptr, 0, 2);
    const double err_w = L2norm(wh, fun_exact_w_ptr, 0, 1);
    const double err_u = L2norm(uh, fun_exact_u_ptr, 0, 1);
    const double err_m = L2norm(mh, fun_exact_m_ptr, Kh);
    const double err_balance = L2norm(balance, fun_zero_ptr, Kh);
    const double max_balance = maxNorm(balance, Kh);
    const double max_cone = maxNorm(fabs(cone), Kh);
    const double mass = integral(Kh, mh, 0);

    if (write_vtk) {
        FunFEM2 pExact(Vex, fun_exact_flux_ptr);
        FunFEM2 wExact(Qex, fun_exact_w_ptr);
        FunFEM2 uExact(Qex, fun_exact_u_ptr);
        FunFEM2 mExact(Qex, fun_exact_m_ptr);

        const std::string prefix = "hdiv_arrival_" + example_name() + "_L" + std::to_string(level);
        Paraview<Mesh2T> writer(Kh, prefix + ".vtk");
        writer.add(ph, "p_h", 0, 2);
        writer.add(pExact, "p_exact", 0, 2);
        writer.add(wh, "w_h", 0, 1);
        writer.add(wExact, "w_exact", 0, 1);
        writer.add(uh, "u_h", 0, 1);
        writer.add(uExact, "u_exact", 0, 1);
        writer.add(mh, "m_h");
        writer.add(mExact, "m_exact", 0, 1);
        writer.add(div_ph, "div_p_h");
        writer.add(balance, "balance_divp_plus_m_minus_f");
        writer.add(cone, "cone_ph_eps_mh_minus_1");
    }

    return {h, err_p, err_w, err_u, err_m, err_balance, max_balance, max_cone, mass};
}

void print_results_2d(const std::vector<RunResult2D> &res) {
    using namespace HdivArrival2D;
    std::cout << "\nExample: " << example_name() << "  eps=" << eps << "\n";
    std::cout << std::left
              << std::setw(11) << "h"
              << std::setw(15) << "err_p"
              << std::setw(10) << "rate"
              << std::setw(15) << "err_w"
              << std::setw(10) << "rate"
              << std::setw(15) << "err_u"
              << std::setw(10) << "rate"
              << std::setw(15) << "err_m"
              << std::setw(10) << "rate"
              << std::setw(15) << "L2 balance"
              << std::setw(15) << "max balance"
              << std::setw(15) << "max cone"
              << std::setw(15) << "mass"
              << "\n";

    for (std::size_t i = 0; i < res.size(); ++i) {
        auto rate = [&](double now, double old, double hnow, double hold) -> double {
            if (i == 0 || now <= 0.0 || old <= 0.0) return 0.0;
            return std::log(now / old) / std::log(hnow / hold);
        };
        const double rp = (i == 0) ? 0.0 : rate(res[i].err_p, res[i-1].err_p, res[i].h, res[i-1].h);
        const double rw = (i == 0) ? 0.0 : rate(res[i].err_w, res[i-1].err_w, res[i].h, res[i-1].h);
        const double ru = (i == 0) ? 0.0 : rate(res[i].err_u, res[i-1].err_u, res[i].h, res[i-1].h);
        const double rm = (i == 0) ? 0.0 : rate(res[i].err_m, res[i-1].err_m, res[i].h, res[i-1].h);

        std::cout << std::left
                  << std::setw(11) << res[i].h
                  << std::setw(15) << res[i].err_p
                  << std::setw(10) << rp
                  << std::setw(15) << res[i].err_w
                  << std::setw(10) << rw
                  << std::setw(15) << res[i].err_u
                  << std::setw(10) << ru
                  << std::setw(15) << res[i].err_m
                  << std::setw(10) << rm
                  << std::setw(15) << res[i].err_balance
                  << std::setw(15) << res[i].max_balance
                  << std::setw(15) << res[i].max_cone
                  << std::setw(15) << res[i].mass
                  << "\n";
    }
}


// -----------------------------------------------------------------------------
// 3D visual example: true unregularized MCF arrival-time translator.
// -----------------------------------------------------------------------------

using Mesh3T    = Mesh3;
using Space3    = FESpace3;
using FunTest3  = TestFunction<Mesh3T>;
using FunFEM3   = FunFEM<Mesh3T>;

namespace HdivArrival3D {

// This example is deliberately not written in the exponentially scaled w-variable.
// The previous finite-eps w-formulation created enormous dynamic ranges on fine
// meshes.  Here we solve the direct arrival-time mixed system for an exact
// mean-convex MCF translator, which is much better conditioned for visual checks.
//
// Exact arrival time:
//      u(x,y,z) = y + log(cos x),      |x| < pi/2.
// Then
//      Du       = (-tan x, 1, 0),
//      |Du|     = sec x,
//      p        = Du/|Du| = (-sin x, cos x, 0),
//      m        = 1/|Du| = cos x,
// and therefore
//      div p + m = -cos x + cos x = 0.
// The level surfaces {u=t} are grim-reaper cylinders.  They are genuine
// mean-convex MCF translators: increasing t translates the same surface in y.

static double x0 = -1.15, lx = 2.30;
static double yorigin = -0.90, ly = 1.80;
static double z0 = -0.65, lz = 1.30;

std::string example_name() { return "grim_reaper_cylinder_true_mcf_direct_u"; }

inline void exact_u_grad(const R3 P,
                         double &u, double &ux, double &uy, double &uz) {
    const double x = P.x;
    u  = P.y + std::log(std::cos(x));
    ux = -std::tan(x);
    uy = 1.0;
    uz = 0.0;
}

R fun_exact_u(const R3 P, int comp, int dom) {
    double u, ux, uy, uz;
    exact_u_grad(P, u, ux, uy, uz);
    return u;
}

R fun_exact_flux(const R3 P, int comp, int dom) {
    // p = Du/|Du|.  Since |Du| = sec x on the chosen strip,
    // this simplifies to (-sin x, cos x, 0).
    const double x = P.x;
    if (comp == 0) return -std::sin(x);
    if (comp == 1) return  std::cos(x);
    return 0.0;
}

R fun_exact_m(const R3 P, int comp, int dom) {
    return std::cos(P.x);
}

R fun_binv(const R3 P, int comp, int dom) {
    // b = 1/|Du| = cos x, p=b Du, so b^{-1}=sec x.
    return 1.0 / std::cos(P.x);
}

R fun_source(const R3 P, int comp, int dom) {
    // div p + m = 0.
    return 0.0;
}

R fun_one(const R3 P, int comp, int dom) { return 1.0; }
R fun_zero(const R3 P, int comp, int dom) { return 0.0; }

inline R3 to_R3(double *P) { return R3(P[0], P[1], P[2]); }

R fun_exact_u_ptr(double *P, int comp) { return fun_exact_u(to_R3(P), comp, 0); }
R fun_exact_flux_ptr(double *P, int comp) { return fun_exact_flux(to_R3(P), comp, 0); }
R fun_exact_m_ptr(double *P, int comp) { return fun_exact_m(to_R3(P), comp, 0); }
R fun_binv_ptr(double *P, int comp) { return fun_binv(to_R3(P), comp, 0); }
R fun_source_ptr(double *P, int comp) { return fun_source(to_R3(P), comp, 0); }
R fun_one_ptr(double *P, int comp) { return 1.0; }
R fun_zero_ptr(double *P, int comp) { return 0.0; }

} // namespace HdivArrival3D

struct RunResult3D {
    double h;
    double err_p;
    double err_u;
    double err_m;
    double err_balance;
    double max_balance;
    double max_unit_flux;
    double mass;
};

RunResult3D run_one_mesh_3d(int nx, int ny, int nz, int level, bool write_vtk) {
    using namespace HdivArrival3D;

    Mesh3T Kh(nx, ny, nz, HdivArrival3D::x0, HdivArrival3D::yorigin, HdivArrival3D::z0,
              HdivArrival3D::lx, HdivArrival3D::ly, HdivArrival3D::lz);
    const double h = lx / static_cast<double>(nx - 1);

    Space3 Vh(Kh, DataFE<Mesh3T>::RT0);
    Space3 Qh(Kh, DataFE<Mesh3T>::P0);

    Lagrange3 FEvecExact(2);
    Space3 Vex(Kh, FEvecExact);
    Space3 Qex(Kh, DataFE<Mesh3T>::P2);

    Normal n;

    FEM<Mesh3T> prob(Vh);
    prob.add(Qh);

    FunFEM3 bInv(Qex, fun_binv_ptr);
    FunFEM3 uD(Qex, fun_exact_u_ptr);
    FunFEM3 mExactField(Qex, fun_exact_m_ptr);
    FunFEM3 zero(Qex, fun_zero_ptr);
    FunFEM3 one(Qex, fun_one_ptr);

    FunTest3 p(Vh, 3), q(Vh, 3);
    FunTest3 u(Qh, 1), v(Qh, 1);

    // Direct unregularized arrival-time mixed method with frozen exact coefficient:
    //      b^{-1} p - grad u = 0,      b=1/|Du_exact|,
    //      div p + m_exact = 0.
    // Weak first equation:
    //      (b^{-1}p,q) + (u,div q) = <u_D, q.n>.
    prob.addBilinear(
        +innerProduct(bInv.expr() * p, q)
        +innerProduct(u, div(q))
        +innerProduct(div(p), v),
        Kh);

    prob.addLinear(+innerProduct(uD.expr(), q * n), Kh, INTEGRAL_BOUNDARY);
    prob.addLinear(-innerProduct(mExactField.expr(), v), Kh);

    prob.solve("umfpack");

    const int ndof_p = Vh.get_nb_dof();
    const int ndof_u = Qh.get_nb_dof();
    Rn_ data_p = prob.rhs_(SubArray(ndof_p, 0));
    Rn_ data_u = prob.rhs_(SubArray(ndof_u, ndof_p));

    FunFEM3 ph(Vh, data_p);
    FunFEM3 uh(Qh, data_u);

    auto div_ph = dx(ph.expr(0)) + dy(ph.expr(1)) + dz(ph.expr(2));
    auto balance = div_ph + mExactField.expr();
    auto unit_flux_defect = ph.expr(0) * ph.expr(0) + ph.expr(1) * ph.expr(1) + ph.expr(2) * ph.expr(2) - one.expr();

    const double err_p = L2norm(ph, fun_exact_flux_ptr, 0, 3);
    const double err_u = L2norm(uh, fun_exact_u_ptr, 0, 1);
    const double err_m = 0.0; // m is prescribed analytically in this direct-u benchmark.
    const double err_balance = L2norm(balance, fun_zero_ptr, Kh);
    const double max_balance = maxNorm(balance, Kh);
    const double max_unit_flux = maxNorm(fabs(unit_flux_defect), Kh);
    const double mass = integral(Kh, mExactField.expr(), 0);

    if (write_vtk) {
        FunFEM3 pExact(Vex, fun_exact_flux_ptr);
        FunFEM3 uExact(Qex, fun_exact_u_ptr);
        FunFEM3 mExact(Qex, fun_exact_m_ptr);

        const std::string prefix = "hdiv_arrival_" + example_name() + "_L" + std::to_string(level);
        Paraview<Mesh3T> writer(Kh, prefix + ".vtk");
        writer.add(ph, "p_h", 0, 3);
        writer.add(pExact, "p_exact", 0, 3);
        writer.add(uh, "u_h", 0, 1);
        writer.add(uExact, "u_exact", 0, 1);
        writer.add(mExact, "m_exact", 0, 1);
        writer.add(div_ph, "div_p_h");
        writer.add(balance, "balance_divp_plus_m");
        writer.add(unit_flux_defect, "unit_flux_defect_ph_squared_minus_1");
    }

    return {h, err_p, err_u, err_m, err_balance, max_balance, max_unit_flux, mass};
}

void print_results_3d(const std::vector<RunResult3D> &res) {
    using namespace HdivArrival3D;
    std::cout << "\nExample: " << example_name() << "\n";
    std::cout << "This is the stable 3D visual example.  Contour u_h in ParaView.\n";
    std::cout << "The initial surface may be taken as u_h=0; later translator surfaces are u_h=t.\n";
    std::cout << "Recommended contour values for u_h: -0.80, -0.50, -0.25, 0.00, 0.25, 0.50, 0.80\n";
    std::cout << std::left
              << std::setw(11) << "h"
              << std::setw(15) << "err_p"
              << std::setw(10) << "rate"
              << std::setw(15) << "err_u"
              << std::setw(10) << "rate"
              << std::setw(15) << "L2 balance"
              << std::setw(15) << "max balance"
              << std::setw(15) << "max |p|^2-1"
              << std::setw(15) << "mass"
              << "\n";

    for (std::size_t i = 0; i < res.size(); ++i) {
        auto rate = [&](double now, double old, double hnow, double hold) -> double {
            if (i == 0 || now <= 0.0 || old <= 0.0) return 0.0;
            return std::log(now / old) / std::log(hnow / hold);
        };
        const double rp = (i == 0) ? 0.0 : rate(res[i].err_p, res[i-1].err_p, res[i].h, res[i-1].h);
        const double ru = (i == 0) ? 0.0 : rate(res[i].err_u, res[i-1].err_u, res[i].h, res[i-1].h);

        std::cout << std::left
                  << std::setw(11) << res[i].h
                  << std::setw(15) << res[i].err_p
                  << std::setw(10) << rp
                  << std::setw(15) << res[i].err_u
                  << std::setw(10) << ru
                  << std::setw(15) << res[i].err_balance
                  << std::setw(15) << res[i].max_balance
                  << std::setw(15) << res[i].max_unit_flux
                  << std::setw(15) << res[i].mass
                  << "\n";
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
#ifdef USE_MPI
    MPIcf cfMPI(argc, argv);
#endif

    const double cpubegin = CPUtime();

    std::vector<int> examples;
    if (argc > 1) {
        examples.push_back(std::atoi(argv[1]));
    } else {
        examples = {0, 1, 2};
    }

    for (int ex : examples) {
        if (ex == 0 || ex == 1) {
            using namespace HdivArrival2D;
            set_example(ex);

            int nx = (example_id == 0) ? 17 : 17;
            int ny = (example_id == 0) ? 33 : 17;
            const int nlevels = 4;

            std::vector<RunResult2D> res;
            for (int lev = 0; lev < nlevels; ++lev) {
                const bool write_vtk = (lev == nlevels - 1);
                res.push_back(run_one_mesh_2d(nx, ny, lev, write_vtk));
                nx = 2 * nx - 1;
                ny = 2 * ny - 1;
            }
            print_results_2d(res);
        } else if (ex == 2) {
            // Keep this intentionally modest: the previous 49x49x41 3D solve was
            // slow and ill-conditioned in the exponential w-variable.  This one
            // gives a quick, checkable 3D MCF visualization.
            int nx = 15, ny = 13, nz = 11;
            const int nlevels = 1;

            std::vector<RunResult3D> res;
            for (int lev = 0; lev < nlevels; ++lev) {
                const bool write_vtk = (lev == nlevels - 1);
                res.push_back(run_one_mesh_3d(nx, ny, nz, lev, write_vtk));
                nx = 2 * nx - 1;
                ny = 2 * ny - 1;
                nz = 2 * nz - 1;
            }
            print_results_3d(res);
        } else {
            std::cout << "Unknown example id " << ex << ". Use 0, 1, or 2.\n";
        }
    }

    std::cout << "\nCPU time: " << CPUtime() - cpubegin << " s\n";
    return 0;
}
