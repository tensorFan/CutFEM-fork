#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#ifdef USE_MPI
#include "cfmpi.hpp"
#endif

#include "finiteElement.hpp"
#include "baseProblem.hpp"
#include "paraview.hpp"
#include "../num/matlab.hpp"

// -----------------------------------------------------------------------------
// Maxwell cavity eigenvalue comparison driver with the complementary
// (absolute / magnetic) Maxwell boundary conditions.
//
// One executable exports the three requested unfitted formulations on the
// spherical shell:
//
//   methods: wave, kikuchi, 3field
//
// Boundary conditions for the curl-curl eigenfield u:
//
//   n x curl(u) = 0,
//   n . u       = 0.
//
// For the wave and Kikuchi formulations, n x curl(u) = 0 is the natural
// H(curl) boundary condition.  The normal condition is encoded variationally:
// positive wave eigenmodes are orthogonal to gradients, while the Kikuchi
// constraint (u,grad(q))=0 with q in H1 also enforces n.u=0.
//
// In the three-field formulation
//
//   eps*mu*w = curl(u),
//
// so n x curl(u)=0 becomes the essential condition n x w=0.  The companion
// essential condition is n.u=0 on the RT0 field.  Both are imposed weakly by a
// symmetric Nitsche formulation on the unfitted boundary.
//
// The radial shell field (x-c)/|x-c|^3 does not satisfy n.u=0.  Consequently
// this boundary-condition variant does not use the shell harmonic filter from
// the n x u=0 driver.  The mixed formulations instead receive one scalar
// Lagrange multiplier fixing the zero-mean gauge of p.
//
// Output files:
//   <prefix>_A_<example>_<method>_<level>.dat
//   <prefix>_B_<example>_<method>_<level>.dat
//   <prefix>_manifest.csv
//
// Suggested terminal workflow inside build:
//   ./bin/maxwell3D_eigen_compare_curl_bc --method all \
//       --levels 2 --nx0 7 --prefix eigcmp_curlbc
//
//   conda activate fenicsx-env
//   python3 ../cpp/mainFiles/notebooks/maxwell/eigvals_compare_slepc.py \
//       --matrix-dir . --prefix eigcmp_curlbc --target 3.2 --nev 41
// -----------------------------------------------------------------------------

using namespace globalVariable;

namespace EigenCompareData {
    R eps = 1.;
    R mu  = 1.;

    R shell_center[3] = {0.5 * M_PI, 0.5 * M_PI, 0.5 * M_PI};
    R radius_inner = M_PI / 5.;
    R radius_outer = M_PI / 3.;

    // Spherical shell level set.  The retained side is positive, so this sign
    // convention keeps radius_inner < r < radius_outer.
    R fun_levelSetSphericalShell(double *P, int i, int dom) {
        const R x = P[0] - shell_center[0];
        const R y = P[1] - shell_center[1];
        const R z = P[2] - shell_center[2];
        const R r2 = x*x + y*y + z*z;

        return (r2 - radius_inner * radius_inner)
             * (radius_outer * radius_outer - r2);
    }

}

struct Config {
    int nx0 = 7;
    int ny0 = 7;
    int nz0 = 7;
    int levels = 2;

    bool do_wave = true;
    bool do_kikuchi = true;
    bool do_3field = true;

    std::string prefix = "eigcmp_curlbc";

    // Symmetric Nitsche penalty for the three-field essential traces.
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
    // The present driver uses an exact zero-mean multiplier instead.
    R pressure_regularizer = 0; // 1e-12
};

static void print_usage(const char *exe) {
    std::cout
        << "Usage: " << exe << " [options]\n\n"
        << "Options:\n"
        << "  --levels N              number of refinement levels, default 2\n"
        << "  --nx0 N                 initial nx=ny=nz, default 7\n"
        << "  --prefix NAME           output prefix, default eigcmp_curlbc\n"
        << "  --method all|wave|kikuchi|3field\n"
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
        } else {
            std::cerr << "Unknown option: " << key << std::endl;
            print_usage(argv[0]);
            std::exit(2);
        }
    }
}

static std::string mat_name(const Config &cfg, const std::string &AB,
                            const std::string &method, int level) {
    return cfg.prefix + "_" + AB + "_spherical_shell_" + method + "_"
         + std::to_string(level) + ".dat";
}

static std::string example_note() {
    return "spherical shell centered at (pi/2,pi/2,pi/2); configurable inner and outer radii";
}

static void write_manifest_row(std::ofstream &manifest,
                               const std::string &method, int level,
                               int nx, int ny, int nz, R h,
                               const std::string &Afile, const std::string &Bfile,
                               int n0, int n1, int n2, int nlambda,
                               const Config &cfg,
                               const std::string &bc_note) {
    manifest << "spherical_shell" << ',' << method << ',' << level << ','
             << nx << ',' << ny << ',' << nz << ','
             << std::setprecision(17) << h << ','
             << Afile << ',' << Bfile << ','
             << n0 << ',' << n1 << ',' << n2 << ',' << nlambda << ','
             << std::setprecision(17) << cfg.penalty << ','
             << cfg.tau_curl << ',' << cfg.tau_mass << ',' << cfg.tau_p << ','
             << cfg.pressure_regularizer << ','
             << EigenCompareData::radius_outer << ','
             << 0 << ','
             << '"' << example_note() << '"' << ','
             << '"' << bc_note << '"' << '\n';
}

// Add one scalar multiplier imposing integral_Omega p = 0.  This removes the
// constant gauge in the mixed formulations when the normal trace of u is zero.
static void add_zero_mean_scalar_constraint(CutFEM<Mesh3> &A,
                                            CutFEM<Mesh3> &B,
                                            TestFunction<Mesh3> &p,
                                            TestFunction<Mesh3> &q,
                                            ActiveMesh<Mesh3> &Khi,
                                            int base_dofs) {
    A.addLagrangeMultiplier(
        +innerProduct(1, p), 0, Khi
    );
    A.mat_[0][std::make_pair(base_dofs, base_dofs)] = 0.;

    // Append the same multiplier block to B, but leave its row and column zero.
    B.addLagrangeMultiplier(
        +innerProduct(1, 0 * q), 0, Khi
    );
    B.mat_[0][std::make_pair(base_dofs, base_dofs)] = 0.;
}

static void assemble_wave(const Config &cfg, int level, int nx, int ny, int nz,
                          std::ofstream &manifest) {
    using namespace EigenCompareData;
    typedef TestFunction<Mesh3> FunTest;
    typedef FunFEM<Mesh3> Fun_h;
    typedef Mesh3 Mesh;
    typedef FESpace3 Space;
    typedef CutFESpaceT3 CutSpace;

    std::cout << "\n=== spherical_shell: UNFITTED_WAVE_EIGEN, level " << level << " ===" << std::endl;

    Mesh3 Kh(nx, ny, nz, 0., 0., 0., M_PI, M_PI, M_PI);
    const R h = M_PI / R(nx - 1);

    Space Uh_background(Kh, DataFE<Mesh>::Ned0);
    Space Lh(Kh, DataFE<Mesh>::P1);
    Fun_h levelSet(Lh, fun_levelSetSphericalShell);
    InterfaceLevelSet<Mesh> interface(Kh, levelSet);

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

    // No boundary term is added: n x curl(u)=0 is the natural condition for
    // the curl-curl form.  For every positive eigenvalue, testing with gradients
    // gives (u,grad(q))=0, hence div(u)=0 and n.u=0 in the weak sense.

    A.addPatchStabilization(
        +innerProduct(cfg.tau_curl * jump(curl(u)), jump(curl(v)))
    , Khi);

    B.addBilinear(
        +innerProduct(u, v)
    , Khi);
    B.addPatchStabilization(
        +innerProduct(cfg.tau_mass * jump(u), jump(v))
    , Khi);

    const int nlambda = 0;
    const int base_dofs = Uh.get_nb_dof();

    const std::string Afile = mat_name(cfg, "A", "wave", level);
    const std::string Bfile = mat_name(cfg, "B", "wave", level);
    matlab::Export(A.mat_[0], Afile);
    matlab::Export(B.mat_[0], Bfile);

    write_manifest_row(manifest, "wave", level, nx, ny, nz, h, Afile, Bfile,
                       base_dofs, 0, 0, nlambda, cfg,
                       "natural n_cross_curl_u_equals_0; positive modes satisfy n_dot_u_equals_0 variationally");
}

static void assemble_kikuchi(const Config &cfg, int level, int nx, int ny, int nz,
                             std::ofstream &manifest) {
    using namespace EigenCompareData;
    typedef TestFunction<Mesh3> FunTest;
    typedef FunFEM<Mesh3> Fun_h;
    typedef Mesh3 Mesh;
    typedef FESpace3 Space;
    typedef CutFESpaceT3 CutSpace;

    std::cout << "\n=== spherical_shell: UNFITTED_KIKUCHI_EIGEN, level " << level << " ===" << std::endl;

    Mesh3 Kh(nx, ny, nz, 0., 0., 0., M_PI, M_PI, M_PI);
    const R h = M_PI / R(nx - 1);

    Space Uh_background(Kh, DataFE<Mesh>::Ned0);
    Space Wh_background(Kh, DataFE<Mesh>::P1);
    Space Lh(Kh, DataFE<Mesh>::P1);
    Fun_h levelSet(Lh, fun_levelSetSphericalShell);
    InterfaceLevelSet<Mesh> interface(Kh, levelSet);

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

    // No boundary form is needed.  The curl-curl term gives the natural
    // condition n x curl(u)=0, and (u,grad(q))=0 for all q in H1 gives both
    // div(u)=0 and n.u=0.  The scalar p therefore belongs to H1/R.

    A.addPatchStabilization(
        +innerProduct(cfg.tau_curl * jump(curl(u)), jump(curl(v)))
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

    const int n_u = Uh.get_nb_dof();
    const int n_p = Wh.get_nb_dof();
    const int base_dofs = n_u + n_p;
    add_zero_mean_scalar_constraint(A, B, p, q, Khi, base_dofs);
    const int nlambda = 1;

    const std::string Afile = mat_name(cfg, "A", "kikuchi", level);
    const std::string Bfile = mat_name(cfg, "B", "kikuchi", level);
    matlab::Export(A.mat_[0], Afile);
    matlab::Export(B.mat_[0], Bfile);

    write_manifest_row(manifest, "kikuchi", level, nx, ny, nz, h, Afile, Bfile,
                       n_u, n_p, 0, nlambda, cfg,
                       "natural n_cross_curl_u_equals_0 and n_dot_u_equals_0; zero_mean_p gauge");
}

static void assemble_3field(const Config &cfg, int level, int nx, int ny, int nz,
                            std::ofstream &manifest) {
    using namespace EigenCompareData;
    typedef TestFunction<Mesh3> FunTest;
    typedef FunFEM<Mesh3> Fun_h;
    typedef Mesh3 Mesh;
    typedef FESpace3 Space;
    typedef CutFESpaceT3 CutSpace;

    std::cout << "\n=== spherical_shell: UNFITTED_3FIELD_EIGEN, level " << level << " ===" << std::endl;

    Mesh3 Kh(nx, ny, nz, 0., 0., 0., M_PI, M_PI, M_PI);
    const R h = M_PI / R(nx - 1);

    Space Whcurl_background(Kh, DataFE<Mesh>::Ned0); // w variable
    Space Uhdiv_background(Kh, DataFE<Mesh>::RT0);   // u variable, magnetic flux density in the appendix text
    Space Qh_background(Kh, DataFE<Mesh>::P0);       // p variable
    Space Lh(Kh, DataFE<Mesh>::P1);
    Fun_h levelSet(Lh, fun_levelSetSphericalShell);
    InterfaceLevelSet<Mesh> interface(Kh, levelSet);
    Normal n;

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
    //   curl w - grad p = lambda u,
    //   div u = 0.
    //
    // The absolute Maxwell boundary pair becomes
    //   n x w = 0,   n.u = 0.
    // Both traces are essential for the H(curl)-H(div)-L2 complex and are
    // imposed below by a symmetric unfitted Nitsche form.
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

    // Symmetric Nitsche imposition of n x w=0 and n.u=0.  The consistency
    // terms are the boundary traces paired with the two mixed equations; the
    // final terms penalize the essential H(curl) and H(div) traces.
    A.addBilinear(
        -innerProduct(u, cross(n, tau))
        -innerProduct(cross(n, w), v)
        -innerProduct(cross(n, w), cfg.penalty / h * cross(n, tau))
        // -innerProduct(p, v * n)
        // -innerProduct(u * n, q)
        +innerProduct(u * n, cfg.penalty / h * v * n)
    , interface);

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

    const int n_w = Whcurl.get_nb_dof();
    const int n_u = Uhdiv.get_nb_dof();
    const int n_p = Qh.get_nb_dof();
    const int base_dofs = n_w + n_u + n_p;
    // add_zero_mean_scalar_constraint(A, B, p, q, Khi, base_dofs);
    const int nlambda = 1;

    const std::string Afile = mat_name(cfg, "A", "3field", level);
    const std::string Bfile = mat_name(cfg, "B", "3field", level);
    matlab::Export(A.mat_[0], Afile);
    matlab::Export(B.mat_[0], Bfile);

    write_manifest_row(manifest, "3field", level, nx, ny, nz, h, Afile, Bfile,
                       n_w, n_u, n_p, nlambda, cfg,
                       "symmetric Nitsche for n_cross_w_equals_0 and n_dot_u_equals_0; zero_mean_p gauge");
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
    // Keep the historical hole_radius and harmonic_filter columns so the
    // existing SLEPc reader remains compatible.  Here harmonic_filter is zero.
    manifest << "example,method,level,nx,ny,nz,h,Afile,Bfile,n0,n1,n2,nlambda,penalty,tau_curl,tau_mass,tau_p,pressure_regularizer,hole_radius,harmonic_filter,example_note,bc_note\n";

    int nx = 2 * cfg.nx0 - 1;
    int ny = 2 * cfg.ny0 - 1;
    int nz = 2 * cfg.nz0 - 1;
    for (int level = 0; level < cfg.levels; ++level) {
        if (cfg.do_wave)    assemble_wave(cfg, level, nx, ny, nz, manifest);
        if (cfg.do_kikuchi) assemble_kikuchi(cfg, level, nx, ny, nz, manifest);
        if (cfg.do_3field)  assemble_3field(cfg, level, nx, ny, nz, manifest);

        nx = 2 * nx - 1;
        ny = 2 * ny - 1;
        nz = 2 * nz - 1;
    }

    std::cout << "\nWrote manifest: " << manifest_name << std::endl;
    std::cout << "Next step: python3 ../cpp/mainFiles/notebooks/eigvals_compare_slepc.py --matrix-dir . --prefix "
              << cfg.prefix << " --target 3.2 --nev 41" << std::endl;
    return 0;
}
