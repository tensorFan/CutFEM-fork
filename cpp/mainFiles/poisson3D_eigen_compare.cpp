#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_MPI
#include "cfmpi.hpp"
#endif
#include "finiteElement.hpp"
#include "baseProblem.hpp"
#include "paraview.hpp"
#include "../num/matlab.hpp"

// Unfitted Dirichlet ball: -Delta u = lambda u, u|Gamma = 0.
// For Delta u = lambda u, negate the reported eigenvalues.
// Like maxwell3D_eigen_compare.cpp, export matrices for an external eigensolver.
// See README_poisson3D_eigen_compare.md for the formulation and reproduction.
namespace {
using Mesh = Mesh3;
using Space = FESpace3;
using CutSpace = CutFESpaceT3;
using Field = FunFEM<Mesh>;
using Test = TestFunction<Mesh>;
constexpr int reference_size = 10;
double radius = 1.;

struct Config {
    int nx0 = 9, levels = 3, quadrature = 7;
    double penalty = 100., tau_stiffness = 1., tau_mass = 1., shift = 0.137;
    std::string mass = "both";
    std::filesystem::path prefix = "poisson_eigcmp";
    bool vtk = false;
};

void usage(const char *exe) {
    std::cout << "Usage: " << exe << " [options]\n"
              << "  --nx0 N              background nodes per axis (default 9)\n"
              << "  --levels N           refinements nx <- 2*nx-1 (default 3)\n"
              << "  --radius R           ball radius (default 1)\n"
              << "  --penalty X          symmetric Nitsche coefficient / h (default 100)\n"
              << "  --tau-stiffness X    gradient-jump patch coefficient (default 1)\n"
              << "  --tau-mass X         value-jump patch coefficient (default 1)\n"
              << "  --mass physical|stabilized|both (default both)\n"
              << "  --shift X            background offset in mesh steps (default 0.137)\n"
              << "  --quadrature 5|7     exact-reference integration degree (default 7)\n"
              << "  --prefix PATH        output prefix (default poisson_eigcmp)\n"
              << "  --vtk                read postprocessed eigenvectors and write cut-domain VTK\n"
              << "  --help\n";
}

Config parse(int argc, char **argv) {
    Config c;
    for (int a = 1; a < argc; ++a) {
        const std::string key = argv[a];
        if (key == "--help" || key == "-h") { usage(argv[0]); std::exit(0); }
        if (key == "--vtk") { c.vtk = true; continue; }
        if (++a == argc) throw std::runtime_error("Missing value after " + key);
        const std::string value = argv[a];
        if (key == "--prefix") { c.prefix = value; continue; }
        if (key == "--mass") { c.mass = value; continue; }
        std::size_t used = 0;
        const double x = std::stod(value, &used);
        if (used != value.size() || !std::isfinite(x))
            throw std::runtime_error("Invalid value after " + key);
        if (key == "--nx0" || key == "--levels" || key == "--quadrature") {
            if (x != std::floor(x) || x > std::numeric_limits<int>::max() || x < 0.)
                throw std::runtime_error("Expected a nonnegative integer after " + key);
            if (key == "--nx0") c.nx0 = static_cast<int>(x);
            if (key == "--levels") c.levels = static_cast<int>(x);
            if (key == "--quadrature") c.quadrature = static_cast<int>(x);
        } else if (key == "--radius") radius = x;
        else if (key == "--penalty") c.penalty = x;
        else if (key == "--tau-stiffness") c.tau_stiffness = x;
        else if (key == "--tau-mass") c.tau_mass = x;
        else if (key == "--shift") c.shift = x;
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (c.nx0 < 5 || c.levels < 1 || c.levels > 8 || radius <= 0. || c.penalty <= 0. ||
        c.tau_stiffness < 0. || c.tau_mass < 0. || std::abs(c.shift) > 0.2 ||
        (c.quadrature != 5 && c.quadrature != 7) ||
        (c.mass != "physical" && c.mass != "stabilized" && c.mass != "both"))
        throw std::runtime_error("Invalid configuration; see --help.");
    return c;
}

double level_set(double *p, int, int) {
    return radius * radius - p[0]*p[0] - p[1]*p[1] - p[2]*p[2];
}

std::vector<std::string> methods(const Config &c) {
    if (c.mass == "both") return {"physical", "stabilized"};
    return {c.mass};
}

std::string name(const Config &c, const std::string &kind, int level,
                 const std::string &method = "p1") {
    return c.prefix.filename().string() + "_" + kind + "_ball_" + method + "_" +
           std::to_string(level) + ".dat";
}

std::filesystem::path path(const Config &c, const std::string &file) {
    return c.prefix.parent_path() / file;
}

std::ofstream output(const std::filesystem::path &file) {
    std::ofstream out(file);
    if (!out) throw std::runtime_error("Cannot write " + file.string());
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::setprecision(17);
    return out;
}

// The first ten eigenfunctions in spectral order: (l,n)=(0,1), (1,1),
// (2,1), (0,2), with multiplicities 1,3,5,1. Real spherical harmonics are
// orthonormal on S^2. The radial factor is normalized analytically on the ball.
// Zeros and spherical Bessel definitions: https://dlmf.nist.gov/10.47 and /10.58.
std::array<double, reference_size> exact_modes(const R3 &p) {
    const double pi = std::acos(-1.);
    const double r = p.norm();
    const double x = r > 0. ? p[0]/r : 0.;
    const double y = r > 0. ? p[1]/r : 0.;
    const double z = r > 0. ? p[2]/r : 0.;
    const double j1root = 4.4934094579090641753, j2root = 5.7634591968945497914;
    auto radial = [&](int l, double root) {
        const double j = r > 0. ? std::sph_bessel(l, root*r/radius) : (l == 0 ? 1. : 0.);
        return std::sqrt(2./(radius*radius*radius)) * j / std::abs(std::sph_bessel(l+1, root));
    };
    const double a0 = radial(0, pi) / std::sqrt(4.*pi);
    const double a1 = radial(1, j1root) * std::sqrt(3./(4.*pi));
    const double a2 = radial(2, j2root);
    const double q = std::sqrt(15./(4.*pi));
    return {a0, a1*x, a1*y, a1*z,
            a2*q*x*y, a2*q*x*z, a2*q*y*z,
            a2*0.5*q*(x*x-y*y), a2*std::sqrt(5./(16.*pi))*(3.*z*z-1.),
            radial(0, 2.*pi)/std::sqrt(4.*pi)};
}

// Integrate analytic functions at physical quadrature points, without first
// interpolating them into P1. F_ij=(phi_i,exact_j)_Omega_h, G_ij=(exact_i,exact_j).
// G uses the same cut domain as M, so principal angles are genuine L2 angles.
double export_references(const Config &c, int level, const ActiveMesh<Mesh> &active,
                         const CutSpace &space) {
    const int ndof = space.get_nb_dof();
    std::vector<std::array<double, reference_size>> loads(ndof);
    double gram[reference_size][reference_size] = {};
    const auto &qf = *QF_Simplex<R3>(c.quadrature);
    double volume = 0.;
    for (int k = active.first_element(); k < active.last_element(); k += active.next_element()) {
        const auto fe = space[k];
        const Cut_Part<Mesh::Element> cut(active.get_cut_part(k, 0));
        for (auto it = cut.element_begin(); it != cut.element_end(); ++it) {
            for (int q = 0; q < qf.getNbrOfQuads(); ++q) {
                const R3 p = cut.mapToPhysicalElement(it, qf[q]);
                const R3 ref = fe.T.mapToReferenceElement(p);
                const double basis[4] = {1.-ref[0]-ref[1]-ref[2], ref[0], ref[1], ref[2]};
                const double w = cut.measure(it) * qf[q].getWeight();
                const auto values = exact_modes(p);
                volume += w;
                for (int j = 0; j < reference_size; ++j) {
                    for (int i = 0; i < 4; ++i) loads[fe(i)][j] += w*basis[i]*values[j];
                    for (int i = 0; i < reference_size; ++i) gram[i][j] += w*values[i]*values[j];
                }
            }
        }
    }
    auto f = output(path(c, name(c, "F", level)));
    for (const auto &row : loads) {
        for (double v : row) f << v << ' ';
        f << '\n';
    }
    auto g = output(path(c, name(c, "G", level)));
    for (const auto &row : gram) {
        for (double v : row) g << v << ' ';
        g << '\n';
    }
    return volume;
}

void write_vtk(const Config &c, int level, double h, const ActiveMesh<Mesh> &active,
               CutSpace &space) {
    for (const auto &method : methods(c)) {
        const auto file = path(c, name(c, "eigenvectors", level, method));
        std::ifstream in(file);
        int rows = 0, columns = 0;
        double saved_h = 0., saved_radius = 0., saved_shift = 0.;
        if (!(in >> rows >> columns >> saved_h >> saved_radius >> saved_shift) ||
            rows != space.get_nb_dof() || columns < 1 || columns > 100 ||
            std::abs(saved_h-h) > 1e-12*h || std::abs(saved_radius-radius) > 1e-12*radius ||
            std::abs(saved_shift-c.shift) > 1e-12)
            throw std::runtime_error("Missing/incompatible eigenvectors: " + file.string());
        std::vector<std::vector<double>> data(columns, std::vector<double>(rows));
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < columns; ++j)
                if (!(in >> data[j][i]) || !std::isfinite(data[j][i]))
                    throw std::runtime_error("Invalid eigenvector data in " + file.string());
        auto vtk = file;
        vtk.replace_extension(".vtk");
        Paraview<Mesh> writer(active, vtk.string(), 17);
        for (int j = 0; j < columns; ++j) {
            Rn coefficients(rows);
            for (int i = 0; i < rows; ++i) coefficients[i] = data[j][i];
            Field field(space, coefficients);
            writer.add(field, "mode_" + std::to_string(j+1), 0, 1);
        }
        std::cout << "Wrote " << vtk << '\n';
    }
}

void compute(const Config &c, int level, int nx, std::ofstream &manifest) {
    const double width = 2.5*radius, h = width/(nx-1);
    // Shift the grid while fixing the physical ball, to avoid systematic cuts
    // through vertices. The P1 quadratic level set defines an inscribed polyhedron.
    Mesh background(nx, nx, nx, -width/2.+c.shift*h, -width/2.+1.3*c.shift*h,
                    -width/2.+1.7*c.shift*h, width, width, width);
    Space p1(background, DataFE<Mesh>::P1);
    Field phi(p1, level_set);
    InterfaceLevelSet<Mesh> interface(background, phi);
    ActiveMesh<Mesh> active(background);
    active.truncate(interface, -1); // retain phi > 0, the ball interior
    CutSpace space(active, p1);
    std::cout << "Level " << level << ": nx=" << nx << ", h=" << h
              << ", DoFs=" << space.get_nb_dof() << std::endl;
    if (c.vtk) { write_vtk(c, level, h, active, space); return; }

    ProblemOption options;
    options.order_space_element_quadrature_ = c.quadrature;
    options.order_space_bord_quadrature_ = c.quadrature;
    CutFEM<Mesh> a(space, options), m(space, options), trace(space, options);
    Test u(space, 1, 0), v(space, 1, 0);
    Normal n;
    a.addBilinear(+innerProduct(grad(u), grad(v)), active);
    // Interface assembly uses -interface.normal(), i.e. -grad(phi)/|grad(phi)|,
    // which is OUTWARD for the retained positive region. No normal sign reversal.
    a.addBilinear(-innerProduct(grad(u)*n, v) - innerProduct(u, grad(v)*n)
                 + innerProduct(c.penalty/h*u, v), interface);
    if (c.tau_stiffness > 0.)
        a.addPatchStabilization(+innerProduct(c.tau_stiffness*jump(grad(u)), jump(grad(v))), active);
    m.addBilinear(+innerProduct(u, v), active);
    trace.addBilinear(+innerProduct(u, v), interface);
    // Patch integrals compare polynomial extensions on K1 union K2, rather than
    // traces on the common face (the latter value jump is zero for continuous P1).
    auto export_matrix = [&](auto &matrix, const std::string &kind, const std::string &method = "p1") {
        const auto file = path(c, name(c, kind, level, method));
        { auto check = output(file); }
        matlab::Export(matrix, file.string());
        if (!std::filesystem::file_size(file)) throw std::runtime_error("Empty matrix: " + file.string());
    };
    export_matrix(a.mat_[0], "A");
    export_matrix(m.mat_[0], "M");
    export_matrix(trace.mat_[0], "T");
    const double volume = export_references(c, level, active, space);
    for (const auto &method : methods(c)) {
        std::string bfile = name(c, "M", level);
        if (method == "stabilized") {
            CutFEM<Mesh> b(space, options);
            b.mat_[0] = m.mat_[0];
            if (c.tau_mass > 0.)
                b.addPatchStabilization(+innerProduct(c.tau_mass*jump(u), jump(v)), active);
            export_matrix(b.mat_[0], "B", method);
            bfile = name(c, "B", level, method);
        }
        manifest << "ball," << method << ',' << level << ',' << nx << ',' << h << ','
                 << radius << ',' << c.shift << ',' << space.get_nb_dof() << ','
                 << c.penalty << ',' << c.tau_stiffness << ','
                 << (method == "physical" ? 0. : c.tau_mass) << ',' << volume << ',' << c.quadrature << ','
                 << name(c, "A", level) << ',' << bfile << ',' << name(c, "M", level) << ','
                 << name(c, "T", level) << ',' << name(c, "F", level) << ',' << name(c, "G", level) << '\n';
    }
}
} // namespace

int main(int argc, char **argv) {
#ifdef USE_MPI
    MPIcf mpi(argc, argv);
    if (MPIcf::size() != 1) {
        if (MPIcf::my_rank() == 0) std::cerr << "Run this matrix exporter with one MPI rank.\n";
        return 2;
    }
#endif
    try {
        const Config c = parse(argc, argv);
        if (!c.prefix.parent_path().empty()) std::filesystem::create_directories(c.prefix.parent_path());
        std::ofstream manifest;
        if (!c.vtk) {
            manifest = output(c.prefix.string() + "_manifest.csv");
            manifest << "example,method,level,nx,h,radius,shift,ndof,penalty,tau_stiffness,tau_mass,volume,quadrature,"
                     << "Afile,Bfile,Mfile,Tfile,Ffile,Gfile\n";
        }
        int nx = c.nx0;
        for (int level = 0; level < c.levels; ++level) {
            compute(c, level, nx, manifest);
            if (nx > (std::numeric_limits<int>::max()-1)/2) throw std::runtime_error("Mesh size overflow.");
            nx = 2*nx-1;
        }
        if (!c.vtk) std::cout << "Wrote " << c.prefix.string() << "_manifest.csv\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "poisson3D_eigen_compare: " << e.what() << '\n';
        return 2;
    }
}
