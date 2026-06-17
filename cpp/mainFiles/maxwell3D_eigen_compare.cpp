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
//   examples: cube, cube_hole, all
//
// Boundary condition:
//   n x u = 0 for the curl-curl unknown.  If u is the electric field, this is
//   the usual PEC condition.  For the magnetic-flux formulation in the appendix
//   text, this is the Dirichlet condition induced by PMC for the original E/H
//   variables.
//
// Important level-set normal convention:
//   In this CutFEM code, the level-set choice
//       phi(x) = |x-c|^2 - R^2
//   gives the opposite normal from the one used in the corresponding fitted
//   single-domain boundary formulas for the punctured box.  Therefore, for the
//   internal spherical hole we use
//       phi_hole(x) = -(|x-c|^2 - R^2 + eps_ls),
//   and keep the negative side.  This both deletes the ball and gives the
//   standard outward normal for Omega = box \ ball on the inner boundary.
//
// Topological example:
//   cube_hole is Omega = [0,pi]^3 \ B_{pi/3}(c), c=(pi/2,pi/2,pi/2).
//   The harmonic vector proxy is
//       h(x) = (x-c)/|x-c|^3,
//   i.e. the formula (x,y,z)/r^3 only after shifting coordinates to the centre
//   of the hole.  By default, h is filtered from the discrete space by one
//   Lagrange multiplier for all three formulations when --example cube_hole is
//   active.  Disable this with --no-harmonic-filter.
//
// Output files:
//   <prefix>_A_<example>_<method>_<level>.dat
//   <prefix>_B_<example>_<method>_<level>.dat
//   <prefix>_manifest.csv
//
// Suggested terminal workflow inside build:
//   ./bin/maxwell3D_eigen_compare --example all --levels 2 --nx0 7 --prefix eigcmp // OR 
//   ./bin/maxwell3D_eigen_compare --example all --method 3field --levels 2 --nx0 7 --prefix eigcmp
//   
//   conda activate fenicsx-env
//   python3 ../cpp/mainFiles/notebooks/eigvals_compare_slepc.py --matrix-dir . --prefix eigcmp --target 3.2 --nev 41
// -----------------------------------------------------------------------------

using namespace globalVariable;

enum class ExampleKind { Cube, CubeHole };

static std::string example_name(ExampleKind ex) {
    if (ex == ExampleKind::Cube) return "cube";
    return "cube_hole";
}

namespace EigenCompareData {
    R eps = 1.;
    R mu  = 1.;

    R hole_radius = M_PI / 3.;
    R hole_center[3] = {0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI};
    R levelset_eps = 1e-12;

    // Unfitted representation of the top face z=pi for the simple cube.
    // The active domain is {phi<0}, i.e. z<pi.
    R fun_levelSetCubeTop(double *P, int i, int dom) {
        return P[2] - M_PI;
    }

    // Unfitted representation of the internal spherical hole.  Since the active
    // side is negative, this keeps the exterior of the ball and deletes the ball.
    // The minus sign is intentional: it matches the outward normal convention on
    // the boundary of the punctured box.
    R fun_levelSetCenteredHole(double *P, int i, int dom) {
        const R x = P[0] - hole_center[0];
        const R y = P[1] - hole_center[1];
        const R z = P[2] - hole_center[2];
        return -(x*x + y*y + z*z - hole_radius*hole_radius + levelset_eps);
    }

    R fun_0(double *P, int i, int dom) {
        return 0.;
    }

    // Harmonic representative for the punctured cube, in centred coordinates.
    // It is smooth on Omega because the ball containing r=0 is removed.
    R fun_harmonic_two_form(double *P, int i, int dom) {
        const R x = P[0] - hole_center[0];
        const R y = P[1] - hole_center[1];
        const R z = P[2] - hole_center[2];
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
    return fun_levelSetCenteredHole;
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
    // comparison.  Use --example cube or --example cube_hole to restrict it.
    std::vector<ExampleKind> examples = {ExampleKind::Cube, ExampleKind::CubeHole};

    std::string prefix = "eigcmp";

    // Symmetric Nitsche penalty for n x u = 0.
    R penalty = 1e2;

    // Ghost penalties.  These are configurable because the three formulations
    // have different unknowns, but the defaults keep runs reproducible.
    R tau_curl = 1e0;
    R tau_mass = 1e0;
    R tau_p    = 1e0;

    R tau_w_3field = 1e0;
    R tau_m_3field = 1e0;
    R tau_b_3field = 1e0;

    // Small pressure mass in mixed generalized eigenproblems, scaled by h^{-3}.
    R pressure_regularizer = 1e-12;

    // Use one global constraint (u,h)_Omega=0 in the punctured box.  For 3field,
    // this is applied to the magnetic-flux / H(div) variable u, not to w.
    bool filter_harmonic_in_hole = true;
};

static bool use_harmonic_filter(const Config &cfg, ExampleKind ex) {
    return cfg.filter_harmonic_in_hole && ex == ExampleKind::CubeHole;
}

static void print_usage(const char *exe) {
    std::cout
        << "Usage: " << exe << " [options]\n\n"
        << "Options:\n"
        << "  --levels N              number of refinement levels, default 2\n"
        << "  --nx0 N                 initial nx=ny=nz, default 7\n"
        << "  --prefix NAME           output prefix, default eigcmp\n"
        << "  --method all|wave|kikuchi|3field\n"
        << "  --example all|cube|cube_hole\n"
        << "  --no-harmonic-filter    do not constrain the harmonic field in cube_hole\n"
        << "  --hole-radius X         internal hole radius, default pi/3\n"
        << "  --penalty X             Nitsche penalty, default 1e2\n"
        << "  --tau-curl X            curl ghost penalty, default 1e-2\n"
        << "  --tau-mass X            mass ghost penalty, default 1e-2\n"
        << "  --tau-p X               scalar pressure ghost penalty, default 1e-2\n"
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
        } else if (key == "--hole-radius") {
            EigenCompareData::hole_radius = std::stod(require_value(key));
        } else if (key == "--no-harmonic-filter") {
            cfg.filter_harmonic_in_hole = false;
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
                cfg.examples.push_back(ExampleKind::CubeHole);
            } else if (ex == "cube") {
                cfg.examples.push_back(ExampleKind::Cube);
            } else if (ex == "cube_hole" || ex == "hole") {
                cfg.examples.push_back(ExampleKind::CubeHole);
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
    return "Omega=[0,pi]^3 minus centred ball; hole level-set is negated for outward normal";
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
             << EigenCompareData::hole_radius << ','
             << (use_harmonic_filter(cfg, ex) ? 1 : 0) << ','
             << '"' << example_note(ex) << '"' << ','
             << '"' << bc_note << '"' << '\n';
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
    Khi.truncate(interface, 1);
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
    A.addBilinear(
        -innerProduct(epsi * mui * curl(u), cross(n, v))
        -innerProduct(epsi * mui * cross(n, u), curl(v))
        +innerProduct(cross(n, u), cfg.penalty / h * cross(n, v))
    , Khi, INTEGRAL_BOUNDARY);

    A.addPatchStabilization(
        +innerProduct(cfg.tau_curl * h * h * jump(curl(u)), jump(curl(v)))
    , Khi);

    B.addBilinear(
        +innerProduct(u, v)
    , Khi);
    B.addPatchStabilization(
        +innerProduct(cfg.tau_mass * jump(u), jump(v))
    , Khi);

    int nlambda = 0;
    const int base_dofs = Uh.get_nb_dof();
    if (use_harmonic_filter(cfg, ex)) {
        Lagrange3 VelocitySpace(2);
        Space Vel_background(Kh, VelocitySpace);
        CutSpace Velh(Khi, Vel_background);
        Fun_h harmonic(Velh, fun_harmonic_two_form);

        A.addLagrangeMultiplier(+innerProduct(harmonic.exprList(), v), 0, Khi);
        A.mat_[0][std::make_pair(base_dofs, base_dofs)] = 0.;

        B.addLagrangeMultiplier(+innerProduct(harmonic.exprList(), 0 * v), 0, Khi);
        B.mat_[0][std::make_pair(base_dofs, base_dofs)] = 0.;
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
    Khi.truncate(interface, 1);
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
        -innerProduct(epsi * mui * curl(u), cross(n, v))
        -innerProduct(epsi * mui * cross(n, u), curl(v))
        +innerProduct(cross(n, u), cfg.penalty / h * cross(n, v))
        +innerProduct(p, cfg.penalty / h * q)
    , interface);
    A.addBilinear(
        -innerProduct(epsi * mui * curl(u), cross(n, v))
        -innerProduct(epsi * mui * cross(n, u), curl(v))
        +innerProduct(cross(n, u), cfg.penalty / h * cross(n, v))
        +innerProduct(p, cfg.penalty / h * q)
    , Khi, INTEGRAL_BOUNDARY);

    A.addPatchStabilization(
        +innerProduct(cfg.tau_curl * h * h * jump(curl(u)), jump(curl(v)))
        +innerProduct(cfg.tau_p * jump(grad(p)), jump(v))
        +innerProduct(cfg.tau_p * jump(u), jump(grad(q)))
        // +innerProduct(cfg.tau_p * jump(p), jump(q))
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
        Lagrange3 VelocitySpace(2);
        Space Vel_background(Kh, VelocitySpace);
        CutSpace Velh(Khi, Vel_background);
        Fun_h harmonic(Velh, fun_harmonic_two_form);

        A.addLagrangeMultiplier(+innerProduct(harmonic.exprList(), v), 0, Khi);
        A.mat_[0][std::make_pair(base_dofs, base_dofs)] = 0.;

        B.addLagrangeMultiplier(+innerProduct(harmonic.exprList(), 0 * v), 0, Khi);
        B.mat_[0][std::make_pair(base_dofs, base_dofs)] = 0.;
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
    Khi.truncate(interface, 1);
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
    // is the natural mixed weak imposition of n x u = 0.  We deliberately do not
    // use the old normal multiplier u.n=0, since that is a different boundary condition.
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
        Lagrange3 VelocitySpace(2);
        Space Vel_background(Kh, VelocitySpace);
        CutSpace Velh(Khi, Vel_background);
        Fun_h harmonic(Velh, fun_harmonic_two_form);

        A.addLagrangeMultiplier(+innerProduct(harmonic.exprList(), v), 0, Khi);
        A.mat_[0][std::make_pair(base_dofs, base_dofs)] = 0.;

        B.addLagrangeMultiplier(+innerProduct(harmonic.exprList(), 0 * v), 0, Khi);
        B.mat_[0][std::make_pair(base_dofs, base_dofs)] = 0.;
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
    manifest << "example,method,level,nx,ny,nz,h,Afile,Bfile,n0,n1,n2,nlambda,penalty,tau_curl,tau_mass,tau_p,pressure_regularizer,hole_radius,harmonic_filter,example_note,bc_note\n";

    for (ExampleKind ex : cfg.examples) {
        int nx = cfg.nx0, ny = cfg.ny0, nz = cfg.nz0;
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
    std::cout << "Next step: python3 eigvals_compare_slepc_v2.py --matrix-dir . --prefix "
              << cfg.prefix << " --target 3.2 --nev 41" << std::endl;
    return 0;
}
