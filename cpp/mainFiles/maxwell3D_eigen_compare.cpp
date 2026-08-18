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
#include "../num/matlab.hpp"

// -----------------------------------------------------------------------------
// Maxwell cavity eigenvalue comparison driver, v2.
//
// One executable exports all requested unfitted formulations and examples:
//
//   methods:  wave, kikuchi, 3field
//   examples: cube, spherical_shell, all
//
// Boundary condition:
//   n x u = 0 for the curl-curl unknown.  If u is the electric field, this is
//   the usual PEC condition.  For the magnetic-flux formulation in the appendix
//   text, this is the Dirichlet condition induced by PMC for the original E/H
//   variables.
//
// Spherical-shell example:
//   Omega = {x : pi/5 < |x-c| < pi/3}, c=(pi/2,pi/2,pi/2).
//   The radial harmonic representative is
//       h(x) = (x-c)/|x-c|^3.
//   By default one Lagrange multiplier filters this mode from every method.
//   The harmonic function is interpolated into the method's u-space: Ned0 for
//   wave/Kikuchi and RT0 for the three-field formulation.  Disable the filter
//   with --no-harmonic-filter.
//
// Output files:
//   <prefix>_A_<example>_<method>_<level>.dat
//   <prefix>_B_<example>_<method>_<level>.dat
//   <prefix>_manifest.csv
//
// Suggested terminal workflow inside build:
//   ./bin/maxwell3D_eigen_compare --example all --levels 2 --nx0 7 --prefix eigcmp // OR 
//   ./bin/maxwell3D_eigen_compare --example all --method 3field --levels 2 --nx0 7 --prefix eigcmp // OR
//   ./bin/maxwell3D_eigen_compare --example spherical_shell --method all --levels 2 --nx0 7 --prefix eigcmp --no-harmonic-filter
//   
//   conda activate fenicsx-env
//   python3 ../cpp/mainFiles/notebooks/maxwell/eigvals_compare_slepc.py --matrix-dir . --prefix eigcmp --target 3.2 --nev 41
// -----------------------------------------------------------------------------

using namespace globalVariable;

enum class ExampleKind { Cube, SphericalShell };

static std::string example_name(ExampleKind ex) {
    if (ex == ExampleKind::Cube) return "cube";
    return "spherical_shell";
}

namespace EigenCompareData {
    R eps = 1.;
    R mu  = 1.;

    R shell_center[3] = {0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI};
    R radius_inner = M_PI / 5.;
    R radius_outer = M_PI / 3.;

    // Unfitted representation of the top face z=pi for the simple cube.
    // The retained side is {phi>0}, i.e. z<pi.
    R fun_levelSetCubeTop(double *P, int i, int dom) {
        return -(P[2] - M_PI);
    }

    // Spherical shell level set.  The retained side is positive, so this sign
    // convention keeps radius_inner < r < radius_outer.
    // On both spherical boundary components the radial harmonic field satisfies
    // n x h = 0.
    R fun_levelSetSphericalShell(double *P, int i, int dom) {
        const R x = P[0] - shell_center[0];
        const R y = P[1] - shell_center[1];
        const R z = P[2] - shell_center[2];
        const R r2 = x*x + y*y + z*z;

        return (r2 - radius_inner * radius_inner)
             * (radius_outer * radius_outer - r2);
    }

    R fun_0(double *P, int i, int dom) {
        return 0.;
    }

    // Radial harmonic representative on the shell, in centred coordinates.
    // It is smooth because r >= radius_inner > 0.
    R fun_harmonic_two_form(double *P, int i, int dom) {
        const R x = P[0] - shell_center[0];
        const R y = P[1] - shell_center[1];
        const R z = P[2] - shell_center[2];
        const R r2 = x*x + y*y + z*z;
        const R r  = std::sqrt(r2);
        const R r3 = r2 * r;
        if (i == 0) return x / r3;
        if (i == 1) return y / r3;
        return z / r3;
    }

    // Kept only for optional visualization/debugging.
    R fun_exact_u(double *P, int i, int dom) {
        const R z = P[2];
        if (i == 0) return std::sin(pi*z);
        return 0.;
    }
}

typedef R (*LevelSetFunction)(double *, int, int);

static LevelSetFunction level_set_function(ExampleKind ex) {
    using namespace EigenCompareData;
    if (ex == ExampleKind::Cube) return fun_levelSetCubeTop;
    return fun_levelSetSphericalShell;
}

struct Config {
    int nx0 = 7;
    int ny0 = 7;
    int nz0 = 7;
    int levels = 2;

    bool do_wave = true;
    bool do_kikuchi = true;
    bool do_3field = true;

    // Default is deliberately all: one terminal command can generate the full
    // comparison.  Use --example cube or --example spherical_shell to restrict it.
    std::vector<ExampleKind> examples = {ExampleKind::Cube, ExampleKind::SphericalShell};

    std::string prefix = "eigcmp";

    // Symmetric Nitsche penalty for n x u = 0.
    R penalty = 1e2;

    // Ghost penalties.  These are configurable because the three formulations
    // have different unknowns, but the defaults keep runs reproducible.
    R tau_curl = 1e0; // 1e0, testade 1e-2 - ingen skillnad!
    R tau_mass = 1e0; // 1e0, testade 1e-2 - ingen skillnad!
    R tau_p    = 1e0; // 1e0, testade 1e-2 - ingen skillnad!

    R tau_w_3field = 1e0;
    R tau_m_3field = 1e0;
    R tau_b_3field = 1e0;

    // Small pressure mass in mixed generalized eigenproblems, scaled by h^{-3}.
    R pressure_regularizer = 0; // 1e-12

    // Use one global constraint (u,h)_Omega=0 on the spherical shell.  For
    // three-field this acts on the magnetic-flux / H(div) variable u, not w.
    bool filter_harmonic_in_shell = true;
};

static bool use_harmonic_filter(const Config &cfg, ExampleKind ex) {
    return cfg.filter_harmonic_in_shell && ex == ExampleKind::SphericalShell;
}

static void print_usage(const char *exe) {
    std::cout
        << "Usage: " << exe << " [options]\n\n"
        << "Options:\n"
        << "  --levels N              number of refinement levels, default 2\n"
        << "  --nx0 N                 initial nx=ny=nz, default 7\n"
        << "  --prefix NAME           output prefix, default eigcmp\n"
        << "  --method all|wave|kikuchi|3field\n"
        << "  --example all|cube|spherical_shell\n"
        << "  --no-harmonic-filter    do not constrain the shell harmonic field\n"
        << "  --inner-radius X        shell inner radius, default pi/5\n"
        << "  --outer-radius X        shell outer radius, default pi/3\n"
        << "  --penalty X             Nitsche penalty, default 1e2\n"
        << "  --tau-curl X            curl ghost penalty, default 1e0\n"
        << "  --tau-mass X            mass ghost penalty, default 1e0\n"
        << "  --tau-p X               scalar pressure ghost penalty, default 1e0\n"
        << "  --help\n";
}

static void parse_args(int argc, char **argv, Config &cfg) {
    for (int a = 1; a < argc; ++a) {
        std::string key(argv[a]);
        auto require_value = [&](const std::string &name) -> std::string {
            if (a + 1 >= argc) {
                std::cerr << "Missing value after " << name << std::endl;
                std::exit(2);
            }
            return std::string(argv[++a]);
        };

        if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (key == "--levels") {
            cfg.levels = std::stoi(require_value(key));
        } else if (key == "--nx0") {
            cfg.nx0 = std::stoi(require_value(key));
            cfg.ny0 = cfg.nx0;
            cfg.nz0 = cfg.nx0;
        } else if (key == "--prefix") {
            cfg.prefix = require_value(key);
        } else if (key == "--penalty") {
            cfg.penalty = std::stod(require_value(key));
        } else if (key == "--tau-curl") {
            cfg.tau_curl = std::stod(require_value(key));
        } else if (key == "--tau-mass") {
            cfg.tau_mass = std::stod(require_value(key));
        } else if (key == "--tau-p") {
            cfg.tau_p = std::stod(require_value(key));
        } else if (key == "--inner-radius") {
            EigenCompareData::radius_inner = std::stod(require_value(key));
        } else if (key == "--outer-radius" || key == "--hole-radius") {
            // --hole-radius is retained as a backward-compatible alias.
            EigenCompareData::radius_outer = std::stod(require_value(key));
        } else if (key == "--no-harmonic-filter") {
            cfg.filter_harmonic_in_shell = false;
        } else if (key == "--method") {
            std::string m = require_value(key);
            cfg.do_wave = cfg.do_kikuchi = cfg.do_3field = false;
            if (m == "all") {
                cfg.do_wave = cfg.do_kikuchi = cfg.do_3field = true;
            } else if (m == "wave") {
                cfg.do_wave = true;
            } else if (m == "kikuchi") {
                cfg.do_kikuchi = true;
            } else if (m == "3field" || m == "threefield") {
                cfg.do_3field = true;
            } else {
                std::cerr << "Unknown method: " << m << std::endl;
                std::exit(2);
            }
        } else if (key == "--example") {
            std::string ex = require_value(key);
            cfg.examples.clear();
            if (ex == "all") {
                cfg.examples.push_back(ExampleKind::Cube);
                cfg.examples.push_back(ExampleKind::SphericalShell);
            } else if (ex == "cube") {
                cfg.examples.push_back(ExampleKind::Cube);
            } else if (ex == "spherical_shell" || ex == "shell" ||
                       ex == "cube_hole" || ex == "hole") {
                // cube_hole/hole are retained as backward-compatible aliases.
                cfg.examples.push_back(ExampleKind::SphericalShell);
            } else {
                std::cerr << "Unknown example: " << ex << std::endl;
                std::exit(2);
            }
        } else {
            std::cerr << "Unknown option: " << key << std::endl;
            print_usage(argv[0]);
            std::exit(2);
        }
    }
}

static std::string mat_name(const Config &cfg, const std::string &AB,
                            ExampleKind ex, const std::string &method, int level) {
    return cfg.prefix + "_" + AB + "_" + example_name(ex) + "_" + method + "_" + std::to_string(level) + ".dat";
}

static std::string example_note(ExampleKind ex) {
    if (ex == ExampleKind::Cube) {
        return "Omega=[0,pi]^3 represented by plane cut z=pi";
    }
    return "spherical shell centered at (pi/2,pi/2,pi/2); configurable inner and outer radii";
}

static void write_manifest_row(std::ofstream &manifest,
                               ExampleKind ex,
                               const std::string &method, int level,
                               int nx, int ny, int nz, R h,
                               const std::string &Afile, const std::string &Bfile,
                               int n0, int n1, int n2, int nlambda,
                               const Config &cfg,
                               const std::string &bc_note) {
    manifest << example_name(ex) << ',' << method << ',' << level << ','
             << nx << ',' << ny << ',' << nz << ','
             << std::setprecision(17) << h << ','
             << Afile << ',' << Bfile << ','
             << n0 << ',' << n1 << ',' << n2 << ',' << nlambda << ','
             << std::setprecision(17) << cfg.penalty << ','
             << cfg.tau_curl << ',' << cfg.tau_mass << ',' << cfg.tau_p << ','
             << cfg.pressure_regularizer << ','
             << EigenCompareData::radius_outer << ','
             << (use_harmonic_filter(cfg, ex) ? 1 : 0) << ','
             << '"' << example_note(ex) << '"' << ','
             << '"' << bc_note << '"' << '\n';
}

// Add the stabilized shell-harmonic constraint using the exact discrete space
// of the eigenfield being filtered.  The caller supplies a temporary CutFEM
// object with the same block layout as A and B.
static void add_shell_harmonic_constraint(CutFEM<Mesh3> &A,
                                          CutFEM<Mesh3> &B,
                                          CutFEM<Mesh3> &lagr,
                                          CutFESpaceT3 &harmonic_space,
                                          TestFunction<Mesh3> &trial,
                                          TestFunction<Mesh3> &test,
                                          ActiveMesh<Mesh3> &Khi,
                                          R h,
                                          R stabilization,
                                          int base_dofs) {
    using namespace EigenCompareData;
    typedef FunFEM<Mesh3> Fun_h;

    // This constructor interpolates the analytic harmonic representative into
    // harmonic_space: Ned0 for wave/Kikuchi and RT0 for three-field.
    Fun_h harmonic(harmonic_space, fun_harmonic_two_form);

    lagr.addLinear(innerProduct(harmonic.exprList(), trial), Khi);
    lagr.addFaceStabilizationRHS(
        +innerProduct(jump(harmonic.exprList()),
                      stabilization * h * jump(trial))
    , Khi);
    Rn lag_row(lagr.rhs_);

    lagr.rhs_ = 0.;
    lagr.addLinear(innerProduct(harmonic.exprList(), test), Khi);
    lagr.addFaceStabilizationRHS(
        +innerProduct(jump(harmonic.exprList()),
                      stabilization * h * jump(test))
    , Khi);

    A.addLagrangeVecToRowAndCol(lag_row, lagr.rhs_, 0);
    A.mat_[0][std::make_pair(base_dofs, base_dofs)] = 0.;

    // Append the same multiplier block to B, but leave its row and column zero.
    B.addLagrangeMultiplier(
        +innerProduct(harmonic.exprList(), 0 * test), 0, Khi
    );
    B.mat_[0][std::make_pair(base_dofs, base_dofs)] = 0.;
}

static void assemble_wave(const Config &cfg, ExampleKind ex, int level, int nx, int ny, int nz,
                          std::ofstream &manifest) {
    using namespace EigenCompareData;
    typedef TestFunction<Mesh3> FunTest;
    typedef FunFEM<Mesh3> Fun_h;
    typedef Mesh3 Mesh;
    typedef FESpace3 Space;
    typedef CutFESpaceT3 CutSpace;

    std::cout << "\n=== " << example_name(ex) << ": UNFITTED_WAVE_EIGEN, level " << level << " ===" << std::endl;

    const R zmax = (ex == ExampleKind::Cube) ? M_PI + 1e-12 : M_PI;
    Mesh3 Kh(nx, ny, nz, 0., 0., 0., M_PI, M_PI, zmax);
    const R h = M_PI / R(nx - 1);

    Space Uh_background(Kh, DataFE<Mesh>::Ned0);
    Space Lh(Kh, DataFE<Mesh>::P1);
    Fun_h levelSet(Lh, level_set_function(ex));
    InterfaceLevelSet<Mesh> interface(Kh, levelSet);
    Normal n;

    ActiveMesh<Mesh> Khi(Kh);
    Khi.truncate(interface, -1); // remove where levelset function is negative
    Khi.info();

    CutSpace Uh(Khi, Uh_background);
    CutFEM<Mesh> A(Uh);
    CutFEM<Mesh> B(Uh);

    FunTest u(Uh, 3, 0), v(Uh, 3, 0);

    const R mui = 1. / mu;
    const R epsi = 1. / eps;

    A.addBilinear(
        +innerProduct(epsi * mui * curl(u), curl(v))
    , Khi);

    // Symmetric H(curl)-Nitsche imposition of n x u = 0.
    A.addBilinear(
        -innerProduct(epsi * mui * curl(u), cross(n, v))
        -innerProduct(epsi * mui * cross(n, u), curl(v))
        +innerProduct(cross(n, u), cfg.penalty / h * cross(n, v))
    , interface);
    if (ex == ExampleKind::Cube) {
    A.addBilinear(
        -innerProduct(epsi * mui * curl(u), cross(n, v))
        -innerProduct(epsi * mui * cross(n, u), curl(v))
        +innerProduct(cross(n, u), cfg.penalty / h * cross(n, v))
    , Khi, INTEGRAL_BOUNDARY);
    }

    A.addPatchStabilization(
        +innerProduct(cfg.tau_curl * jump(curl(u)), jump(curl(v)))
    , Khi);

    // No boundary term is added on B. Nitsche works differently than fitted imposed BC.
    B.addBilinear(
        +innerProduct(u, v)
    , Khi);
    B.addPatchStabilization(
        +innerProduct(cfg.tau_mass * jump(u), jump(v))
    , Khi);

    int nlambda = 0;
    const int base_dofs = Uh.get_nb_dof();
    if (use_harmonic_filter(cfg, ex)) {
        CutFEM<Mesh> lagr(Uh);
        add_shell_harmonic_constraint(
            A, B, lagr, Uh, u, v, Khi, h, cfg.tau_mass, base_dofs
        );
        nlambda = 1;
    }

    const std::string Afile = mat_name(cfg, "A", ex, "wave", level);
    const std::string Bfile = mat_name(cfg, "B", ex, "wave", level);
    matlab::Export(A.mat_[0], Afile);
    matlab::Export(B.mat_[0], Bfile);

    write_manifest_row(manifest, ex, "wave", level, nx, ny, nz, h, Afile, Bfile,
                       base_dofs, 0, 0, nlambda, cfg,
                       "symmetric Hcurl Nitsche for n_cross_u_equals_0");
}

static void assemble_kikuchi(const Config &cfg, ExampleKind ex, int level, int nx, int ny, int nz,
                             std::ofstream &manifest) {
    using namespace EigenCompareData;
    typedef TestFunction<Mesh3> FunTest;
    typedef FunFEM<Mesh3> Fun_h;
    typedef Mesh3 Mesh;
    typedef FESpace3 Space;
    typedef CutFESpaceT3 CutSpace;

    std::cout << "\n=== " << example_name(ex) << ": UNFITTED_KIKUCHI_EIGEN, level " << level << " ===" << std::endl;

    const R zmax = (ex == ExampleKind::Cube) ? M_PI + 1e-12 : M_PI;
    Mesh3 Kh(nx, ny, nz, 0., 0., 0., M_PI, M_PI, zmax);
    const R h = M_PI / R(nx - 1);

    Space Uh_background(Kh, DataFE<Mesh>::Ned0);
    Space Wh_background(Kh, DataFE<Mesh>::P1);
    Space Lh(Kh, DataFE<Mesh>::P1);
    Fun_h levelSet(Lh, level_set_function(ex));
    InterfaceLevelSet<Mesh> interface(Kh, levelSet);
    Normal n;

    ActiveMesh<Mesh> Khi(Kh);
    Khi.truncate(interface, -1);
    Khi.info();

    CutSpace Uh(Khi, Uh_background);
    CutSpace Wh(Khi, Wh_background);

    CutFEM<Mesh> A(Uh); A.add(Wh);
    CutFEM<Mesh> B(Uh); B.add(Wh);

    FunTest u(Uh, 3, 0), v(Uh, 3, 0);
    FunTest p(Wh, 1, 0), q(Wh, 1, 0);

    const R mui = 1. / mu;
    const R epsi = 1. / eps;

    A.addBilinear(
        +innerProduct(epsi * mui * curl(u), curl(v))
        +innerProduct(grad(p), v)
        +innerProduct(u, grad(q))
        +innerProduct(0 * p, q)
    , Khi);

    A.addBilinear(
        -innerProduct(u*n, q)
        -innerProduct(p, v*n)
        
        -innerProduct(epsi * mui * curl(u), cross(n, v))
        -innerProduct(epsi * mui * cross(n, u), curl(v))
        +innerProduct(cross(n, u), cfg.penalty / h * cross(n, v))
        +innerProduct(p, cfg.penalty / h * q)
    , interface);
    if (ex == ExampleKind::Cube) {
    A.addBilinear(
        -innerProduct(u*n, q)
        -innerProduct(p, v*n)

        -innerProduct(epsi * mui * curl(u), cross(n, v))
        -innerProduct(epsi * mui * cross(n, u), curl(v))
        +innerProduct(cross(n, u), cfg.penalty / h * cross(n, v))
        +innerProduct(p, cfg.penalty / h * q)
    , Khi, INTEGRAL_BOUNDARY);
    }

    A.addPatchStabilization(
        +innerProduct(cfg.tau_curl * jump(curl(u)), jump(curl(v)))
        +innerProduct(cfg.tau_p * jump(grad(p)), jump(v))
        +innerProduct(cfg.tau_p * jump(u), jump(grad(q)))
    , Khi);

    // const R regularizer = cfg.pressure_regularizer / (h * h * h);
    B.addBilinear(
        +innerProduct(u, v)
        +innerProduct(p, 0 * q)
        // +innerProduct(regularizer * p, q)
    , Khi);
    B.addPatchStabilization(
        +innerProduct(cfg.tau_mass * jump(u), jump(v))
    , Khi);

    int nlambda = 0;
    const int n_u = Uh.get_nb_dof();
    const int n_p = Wh.get_nb_dof();
    const int base_dofs = n_u + n_p;
    if (use_harmonic_filter(cfg, ex)) {
        CutFEM<Mesh> lagr(Uh); lagr.add(Wh);
        add_shell_harmonic_constraint(
            A, B, lagr, Uh, u, v, Khi, h, cfg.tau_mass, base_dofs
        );
        nlambda = 1;
    }

    const std::string Afile = mat_name(cfg, "A", ex, "kikuchi", level);
    const std::string Bfile = mat_name(cfg, "B", ex, "kikuchi", level);
    matlab::Export(A.mat_[0], Afile);
    matlab::Export(B.mat_[0], Bfile);

    write_manifest_row(manifest, ex, "kikuchi", level, nx, ny, nz, h, Afile, Bfile,
                       n_u, n_p, 0, nlambda, cfg,
                       "symmetric Hcurl Nitsche for n_cross_u_equals_0 plus scalar p boundary penalty");
}

static void assemble_3field(const Config &cfg, ExampleKind ex, int level, int nx, int ny, int nz,
                            std::ofstream &manifest) {
    using namespace EigenCompareData;
    typedef TestFunction<Mesh3> FunTest;
    typedef FunFEM<Mesh3> Fun_h;
    typedef Mesh3 Mesh;
    typedef FESpace3 Space;
    typedef CutFESpaceT3 CutSpace;

    std::cout << "\n=== " << example_name(ex) << ": UNFITTED_3FIELD_EIGEN, level " << level << " ===" << std::endl;

    const R zmax = (ex == ExampleKind::Cube) ? M_PI + 1e-12 : M_PI;
    Mesh3 Kh(nx, ny, nz, 0., 0., 0., M_PI, M_PI, zmax);
    const R h = M_PI / R(nx - 1);

    Space Whcurl_background(Kh, DataFE<Mesh>::Ned0); // w variable
    Space Uhdiv_background(Kh, DataFE<Mesh>::RT0);   // u variable, magnetic flux density in the appendix text
    Space Qh_background(Kh, DataFE<Mesh>::P0);       // p variable
    Space Lh(Kh, DataFE<Mesh>::P1);
    Fun_h levelSet(Lh, level_set_function(ex));
    InterfaceLevelSet<Mesh> interface(Kh, levelSet);

    ActiveMesh<Mesh> Khi(Kh);
    Khi.truncate(interface, -1);
    Khi.info();

    CutSpace Whcurl(Khi, Whcurl_background);
    CutSpace Uhdiv(Khi, Uhdiv_background);
    CutSpace Qh(Khi, Qh_background);

    CutFEM<Mesh> A(Whcurl); A.add(Uhdiv); A.add(Qh);
    CutFEM<Mesh> B(Whcurl); B.add(Uhdiv); B.add(Qh);

    FunTest w(Whcurl, 3, 0), tau(Whcurl, 3, 0);
    FunTest u(Uhdiv, 3, 0), v(Uhdiv, 3, 0);
    FunTest p(Qh, 1, 0), q(Qh, 1, 0);

    // First-order system:
    //   eps*mu*w = curl u,
    //   curl w + grad p = lambda u,
    //   div u = 0.
    // In the first equation the boundary term is (n x u) dot tau.  Dropping it
    // is the natural mixed weak imposition of n x u = 0. The other boundary condition is p=0
    // which appears on integrating by parts the second equation.
    A.addBilinear(
        -innerProduct(eps * mu * w, tau)
        +innerProduct(u, curl(tau))
    , Khi);
    A.addBilinear(
        +innerProduct(curl(w), v)
        +innerProduct(p, div(v))
    , Khi);
    A.addBilinear(
        +innerProduct(div(u), q)
    , Khi);
    A.addBilinear(
        +innerProduct(u, 0 * v)
        +innerProduct(p, 0 * q)
    , Khi);

    A.addPatchStabilization(
        -innerProduct(cfg.tau_w_3field * jump(w), jump(tau))
        +innerProduct(cfg.tau_m_3field * jump(curl(w)), jump(v))
        +innerProduct(cfg.tau_m_3field * jump(u), jump(curl(tau)))
        +innerProduct(cfg.tau_b_3field * jump(p), jump(div(v)))
        +innerProduct(cfg.tau_b_3field * jump(div(u)), jump(q))
    , Khi);

    // const R regularizer = cfg.pressure_regularizer / (h * h * h);
    B.addBilinear(
        +innerProduct(u, v)
        +innerProduct(w, 0 * tau)
        +innerProduct(p, 0 * q)
        // +innerProduct(regularizer * p, q)
    , Khi);
    B.addPatchStabilization(
        +innerProduct(cfg.tau_mass * jump(u), jump(v))
    , Khi);

    int nlambda = 0;
    const int n_w = Whcurl.get_nb_dof();
    const int n_u = Uhdiv.get_nb_dof();
    const int n_p = Qh.get_nb_dof();
    const int base_dofs = n_w + n_u + n_p;
    if (use_harmonic_filter(cfg, ex)) {
        CutFEM<Mesh> lagr(Whcurl); lagr.add(Uhdiv); lagr.add(Qh);
        add_shell_harmonic_constraint(
            A, B, lagr, Uhdiv, u, v, Khi, h,
            cfg.tau_m_3field, base_dofs
        );
        nlambda = 1;
    }

    const std::string Afile = mat_name(cfg, "A", ex, "3field", level);
    const std::string Bfile = mat_name(cfg, "B", ex, "3field", level);
    matlab::Export(A.mat_[0], Afile);
    matlab::Export(B.mat_[0], Bfile);

    write_manifest_row(manifest, ex, "3field", level, nx, ny, nz, h, Afile, Bfile,
                       n_w, n_u, n_p, nlambda, cfg,
                       "natural mixed weak imposition of n_cross_u_equals_0; optional harmonic constraint on u");
}

int main(int argc, char **argv) {
#ifdef USE_MPI
    MPIcf cfMPI(argc, argv);
#endif
    Config cfg;
    parse_args(argc, argv, cfg);

    const std::string manifest_name = cfg.prefix + "_manifest.csv";
    std::ofstream manifest(manifest_name.c_str());
    if (!manifest) {
        std::cerr << "Could not open manifest for writing: " << manifest_name << std::endl;
        return 1;
    }
    // Keep the historical hole_radius column name so the existing SLEPc
    // reader remains compatible; for spherical_shell it stores radius_outer.
    manifest << "example,method,level,nx,ny,nz,h,Afile,Bfile,n0,n1,n2,nlambda,penalty,tau_curl,tau_mass,tau_p,pressure_regularizer,hole_radius,harmonic_filter,example_note,bc_note\n";

    for (ExampleKind ex : cfg.examples) {
        int nx = cfg.nx0, ny = cfg.ny0, nz = cfg.nz0;
        nx = (ex == ExampleKind::Cube) ? nx : 2*nx - 1;
        ny = (ex == ExampleKind::Cube) ? ny : 2*ny - 1;
        nz = (ex == ExampleKind::Cube) ? nz : 2*nz - 1;
        for (int level = 0; level < cfg.levels; ++level) {
            if (cfg.do_wave)    assemble_wave(cfg, ex, level, nx, ny, nz, manifest);
            if (cfg.do_kikuchi) assemble_kikuchi(cfg, ex, level, nx, ny, nz, manifest);
            if (cfg.do_3field)  assemble_3field(cfg, ex, level, nx, ny, nz, manifest);

            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
        }
    }

    std::cout << "\nWrote manifest: " << manifest_name << std::endl;
    std::cout << "Next step: python3 ../cpp/mainFiles/notebooks/eigvals_compare_slepc.py --matrix-dir . --prefix "
              << cfg.prefix << " --target 3.2 --nev 41" << std::endl;
    return 0;
}
