#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_MPI
#include "cfmpi.hpp"
#endif

#include "finiteElement.hpp"
#include "baseProblem.hpp"
#include "paraview.hpp"

namespace scalar_obstacle {

using Mesh = Mesh2;
using Space = FESpace2;
using Problem = FEM<Mesh>;
using Fun = FunFEM<Mesh>;
using Test = TestFunction<Mesh>;

constexpr double PI = 3.141592653589793238462643383279502884;

// -----------------------------------------------------------------------------
// Scalar P2 plus one cubic interior bubble.  This is the scalar analogue of the
// P2+B displacement space used with P0 multipliers in Gustafsson's algorithm.
// The archive contains P2 and a vectorial P2BR element, but not a scalar P2B
// element, so the required scalar bubble enrichment is defined locally here.
// -----------------------------------------------------------------------------
class TypeOfFE_P2BubbleLagrange2d : public GTypeOfFE<Mesh2> {
    using MeshT = Mesh2;
    using Element = typename MeshT::Element;

  public:
    static int Data[];
    static double alpha_Pi_h[];

    TypeOfFE_P2BubbleLagrange2d() : GTypeOfFE<Mesh2>(7, 1, Data, 7, 7, alpha_Pi_h) {
        GTypeOfFE<MeshT>::basisFctType = BasisFctType::P2;
        GTypeOfFE<MeshT>::polynomialOrder = 3;

        static const R2 Pt[7] = {
            R2(0., 0.), R2(1., 0.), R2(0., 1.),
            R2(0.5, 0.5), R2(0., 0.5), R2(0.5, 0.),
            R2(1. / 3., 1. / 3.)
        };
        for (int i = 0; i < 7; ++i) {
            Pt_Pi_h[i] = Pt[i];
            ipj_Pi_h[i] = IPJ(i, i, 0);
        }
    }

    void FB(const What_d whatd, const Element& K, const R2& P, RNMK_& val) const override {
        const double l[3] = {1. - P.x - P.y, P.x, P.y};
        assert(val.N() >= 7);
        assert(val.M() == 1);
        val = 0.;

        R2 Dl[3];
        K.Gradlambda(Dl);

        double p2[6] = {};
        double p2x[6] = {}, p2y[6] = {};
        double p2xx[6] = {}, p2yy[6] = {}, p2xy[6] = {};

        int kk = 0;
        for (int i = 0; i < 3; ++i, ++kk) {
            p2[kk] = l[i] * (2. * l[i] - 1.);
            p2x[kk] = Dl[i].x * (4. * l[i] - 1.);
            p2y[kk] = Dl[i].y * (4. * l[i] - 1.);
            p2xx[kk] = 4. * Dl[i].x * Dl[i].x;
            p2yy[kk] = 4. * Dl[i].y * Dl[i].y;
            p2xy[kk] = 4. * Dl[i].x * Dl[i].y;
        }
        for (int e = 0; e < 3; ++e, ++kk) {
            const int i0 = Element::nvedge[e][0];
            const int i1 = Element::nvedge[e][1];
            p2[kk] = 4. * l[i0] * l[i1];
            p2x[kk] = 4. * (Dl[i1].x * l[i0] + Dl[i0].x * l[i1]);
            p2y[kk] = 4. * (Dl[i1].y * l[i0] + Dl[i0].y * l[i1]);
            p2xx[kk] = 8. * Dl[i0].x * Dl[i1].x;
            p2yy[kk] = 8. * Dl[i0].y * Dl[i1].y;
            p2xy[kk] = 4. * (Dl[i0].x * Dl[i1].y + Dl[i1].x * Dl[i0].y);
        }

        const double bubble = 27. * l[0] * l[1] * l[2];
        const double bubble_x = 27. * (Dl[0].x * l[1] * l[2] + l[0] * Dl[1].x * l[2] + l[0] * l[1] * Dl[2].x);
        const double bubble_y = 27. * (Dl[0].y * l[1] * l[2] + l[0] * Dl[1].y * l[2] + l[0] * l[1] * Dl[2].y);
        const double bubble_xx = 54. * (Dl[0].x * Dl[1].x * l[2] + Dl[0].x * Dl[2].x * l[1] + Dl[1].x * Dl[2].x * l[0]);
        const double bubble_yy = 54. * (Dl[0].y * Dl[1].y * l[2] + Dl[0].y * Dl[2].y * l[1] + Dl[1].y * Dl[2].y * l[0]);
        const double bubble_xy = 27. * (
            Dl[0].x * (Dl[1].y * l[2] + l[1] * Dl[2].y) +
            Dl[1].x * (Dl[0].y * l[2] + l[0] * Dl[2].y) +
            Dl[2].x * (Dl[0].y * l[1] + l[0] * Dl[1].y));

        const double p2_at_bary[6] = {-1. / 9., -1. / 9., -1. / 9., 4. / 9., 4. / 9., 4. / 9.};

        if (whatd & Fop_D0) {
            RN_ f(val('.', 0, op_id));
            for (int i = 0; i < 6; ++i) f[i] = p2[i] - p2_at_bary[i] * bubble;
            f[6] = bubble;
        }
        if (whatd & (Fop_D1 | Fop_D2)) {
            RN_ fx(val('.', 0, op_dx));
            RN_ fy(val('.', 0, op_dy));
            for (int i = 0; i < 6; ++i) {
                fx[i] = p2x[i] - p2_at_bary[i] * bubble_x;
                fy[i] = p2y[i] - p2_at_bary[i] * bubble_y;
            }
            fx[6] = bubble_x;
            fy[6] = bubble_y;

            if (whatd & Fop_D2) {
                RN_ fxx(val('.', 0, op_dxx));
                RN_ fyy(val('.', 0, op_dyy));
                RN_ fxy(val('.', 0, op_dxy));
                for (int i = 0; i < 6; ++i) {
                    fxx[i] = p2xx[i] - p2_at_bary[i] * bubble_xx;
                    fyy[i] = p2yy[i] - p2_at_bary[i] * bubble_yy;
                    fxy[i] = p2xy[i] - p2_at_bary[i] * bubble_xy;
                }
                fxx[6] = bubble_xx;
                fyy[6] = bubble_yy;
                fxy[6] = bubble_xy;
            }
        }
    }
};

int TypeOfFE_P2BubbleLagrange2d::Data[] = {
    0, 1, 2, 3, 4, 5, 6, // support item number of each dof
    0, 0, 0, 0, 0, 0, 0, // number of dof on that item
    0, 1, 2, 3, 4, 5, 6, // interpolation node of each dof
    0, 1, 2, 3, 4, 5, 6, // df in sub FE
    1, 1, 1, 0,          // one vertex dof, one edge dof, one cell dof
    0,                   // component 0 uses this sub FE
    0,                   // begin dof component 0
    7                    // end dof component 0
};

double TypeOfFE_P2BubbleLagrange2d::alpha_Pi_h[] = {1., 1., 1., 1., 1., 1., 1.};

// -----------------------------------------------------------------------------
// Problem data: the membrane-contact benchmark of the paper.
// -----------------------------------------------------------------------------
inline double obstacle_g(const R2& P) {
    return std::sin(PI * P.x) * std::sin(PI * P.y) - 0.5;
}
inline double obstacle_gx(const R2& P) {
    return PI * std::cos(PI * P.x) * std::sin(PI * P.y);
}
inline double obstacle_gy(const R2& P) {
    return PI * std::sin(PI * P.x) * std::cos(PI * P.y);
}
inline double rhs_f(const R2&) { return 0.0; }

double fun_g(double* P, int) { return std::sin(PI * P[0]) * std::sin(PI * P[1]) - 0.5; }
double fun_f(double*, int) { return 0.0; }
double fun_zero(double*, int) { return 0.0; }

struct SimpleMesh {
    std::vector<R2> vertices;
    std::vector<std::array<int, 3>> triangles;
};

struct Options {
    int initial_nx = 9;
    int adaptive_steps = 6;
    int max_pd_iterations = 30;
    int vtk_every = 1;
    double active_tol = 1e-12;
    double pd_tol = 1e-10;
    double marking_beta = 0.5;
    std::string output_prefix = "scalar_obstacle";
    std::string solver_name = "default";
};

struct MeshResult {
    int level = 0;
    int ndof = 0;
    int ntri = 0;
    int active = 0;
    int pd_iterations = 0;
    double pd_residual = 0.;
    double eta_total = 0.;
    double eta_int = 0.;
    double eta_jump = 0.;
    double eta_contact = 0.;
    double max_violation = 0.;
    double complementarity = 0.;
    std::vector<double> eta_cell;
    std::vector<double> lambda_cell;
    std::vector<double> active_cell;
};

SimpleMesh make_initial_mesh(int nx, int ny) {
    SimpleMesh m;
    m.vertices.reserve(nx * ny);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            m.vertices.emplace_back(double(i) / double(nx - 1), double(j) / double(ny - 1));
        }
    }
    auto id = [nx](int i, int j) { return i + j * nx; };
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            const int v0 = id(i, j);
            const int v1 = id(i + 1, j);
            const int v2 = id(i, j + 1);
            const int v3 = id(i + 1, j + 1);
            m.triangles.push_back({v0, v1, v2});
            m.triangles.push_back({v3, v2, v1});
        }
    }
    return m;
}

inline double signed_area(const R2& A, const R2& B, const R2& C) {
    return 0.5 * ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x));
}

int boundary_label(const R2& A, const R2& B) {
    const double tol = 1e-12;
    if (std::abs(A.y) < tol && std::abs(B.y) < tol) return 1;
    if (std::abs(A.x - 1.) < tol && std::abs(B.x - 1.) < tol) return 2;
    if (std::abs(A.y - 1.) < tol && std::abs(B.y - 1.) < tol) return 3;
    if (std::abs(A.x) < tol && std::abs(B.x) < tol) return 4;
    return 0;
}

int vertex_label(const R2& P) {
    const double tol = 1e-12;
    int lab = 0;
    if (std::abs(P.y) < tol) lab = std::max(lab, 1);
    if (std::abs(P.x - 1.) < tol) lab = std::max(lab, 2);
    if (std::abs(P.y - 1.) < tol) lab = std::max(lab, 3);
    if (std::abs(P.x) < tol) lab = std::max(lab, 4);
    return lab;
}

using EdgeKey = std::pair<int, int>;
inline EdgeKey edge_key(int a, int b) { return (a < b) ? EdgeKey(a, b) : EdgeKey(b, a); }

void add_oriented_triangle(SimpleMesh& out, int a, int b, int c) {
    if (signed_area(out.vertices[a], out.vertices[b], out.vertices[c]) < 0.) std::swap(b, c);
    out.triangles.push_back({a, b, c});
}

void write_mesh_file(const SimpleMesh& mesh, const std::string& filename) {
    std::map<EdgeKey, int> count;
    const int e2v[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    for (const auto& T : mesh.triangles) {
        for (auto& e : e2v) ++count[edge_key(T[e[0]], T[e[1]])];
    }
    std::vector<std::array<int, 3>> bedges;
    for (const auto& kv : count) {
        if (kv.second == 1) {
            const int a = kv.first.first, b = kv.first.second;
            const int lab = boundary_label(mesh.vertices[a], mesh.vertices[b]);
            if (lab != 0) bedges.push_back({a, b, lab});
        }
    }

    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot write mesh file " + filename);
    out << mesh.vertices.size() << ' ' << mesh.triangles.size() << ' ' << bedges.size() << "\n";
    out << std::setprecision(17);
    for (const auto& P : mesh.vertices) out << P.x << ' ' << P.y << ' ' << vertex_label(P) << "\n";
    for (const auto& T : mesh.triangles) out << T[0] + 1 << ' ' << T[1] + 1 << ' ' << T[2] + 1 << " 0\n";
    for (const auto& E : bedges) out << E[0] + 1 << ' ' << E[1] + 1 << ' ' << E[2] << "\n";
}

SimpleMesh refine_marked(const SimpleMesh& in, const std::vector<bool>& marked) {
    const int e2v[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    std::set<EdgeKey> split_edges;
    for (std::size_t k = 0; k < in.triangles.size(); ++k) {
        if (!marked[k]) continue;
        const auto& T = in.triangles[k];
        for (auto& e : e2v) split_edges.insert(edge_key(T[e[0]], T[e[1]]));
    }

    SimpleMesh out;
    out.vertices = in.vertices;
    std::map<EdgeKey, int> midpoint;
    auto get_midpoint = [&](int a, int b) -> int {
        const EdgeKey key = edge_key(a, b);
        auto it = midpoint.find(key);
        if (it != midpoint.end()) return it->second;
        const R2 P(0.5 * (out.vertices[a].x + out.vertices[b].x), 0.5 * (out.vertices[a].y + out.vertices[b].y));
        const int idx = static_cast<int>(out.vertices.size());
        out.vertices.push_back(P);
        midpoint[key] = idx;
        return idx;
    };

    for (const auto& T : in.triangles) {
        bool split[3] = {};
        int mid[3] = {-1, -1, -1};
        int ns = 0;
        for (int e = 0; e < 3; ++e) {
            if (split_edges.count(edge_key(T[e2v[e][0]], T[e2v[e][1]]))) {
                split[e] = true;
                mid[e] = get_midpoint(T[e2v[e][0]], T[e2v[e][1]]);
                ++ns;
            }
        }
        if (ns == 0) {
            add_oriented_triangle(out, T[0], T[1], T[2]);
        } else if (ns == 3) {
            const int m01 = mid[0], m12 = mid[1], m20 = mid[2];
            add_oriented_triangle(out, T[0], m01, m20);
            add_oriented_triangle(out, T[1], m12, m01);
            add_oriented_triangle(out, T[2], m20, m12);
            add_oriented_triangle(out, m01, m12, m20);
        } else if (ns == 1) {
            int e = split[0] ? 0 : (split[1] ? 1 : 2);
            const int a = T[e2v[e][0]], b = T[e2v[e][1]], m = mid[e];
            const int c = T[3 - e2v[e][0] - e2v[e][1]];
            add_oriented_triangle(out, a, m, c);
            add_oriented_triangle(out, m, b, c);
        } else { // ns == 2
            int common = -1, b = -1, c = -1;
            int mab = -1, mac = -1;
            for (int v = 0; v < 3; ++v) {
                int incident = 0;
                for (int e = 0; e < 3; ++e) {
                    if (split[e] && (T[e2v[e][0]] == T[v] || T[e2v[e][1]] == T[v])) ++incident;
                }
                if (incident == 2) { common = T[v]; break; }
            }
            std::vector<int> others;
            for (int v = 0; v < 3; ++v) if (T[v] != common) others.push_back(T[v]);
            b = others[0]; c = others[1];
            mab = get_midpoint(common, b);
            mac = get_midpoint(common, c);
            add_oriented_triangle(out, common, mab, mac);
            add_oriented_triangle(out, mab, b, c);
            add_oriented_triangle(out, mab, c, mac);
        }
    }
    return out;
}

void constrain_dof(std::map<std::pair<int, int>, R>& A, Rn& b, int dof, double value = 0.) {
    for (auto it = A.begin(); it != A.end();) {
        if (it->first.first == dof || it->first.second == dof) it = A.erase(it);
        else ++it;
    }
    A[std::make_pair(dof, dof)] = 1.;
    b(dof) = value;
}

std::vector<double> cell_average_u_minus_g(const Mesh& Th, Fun& uh) {
    static const double bary[3][3] = {
        {2. / 3., 1. / 6., 1. / 6.}, {1. / 6., 2. / 3., 1. / 6.}, {1. / 6., 1. / 6., 2. / 3.}
    };
    std::vector<double> avg(Th.nt, 0.);
    for (int k = Th.first_element(); k < Th.last_element(); k += Th.next_element()) {
        const auto& K = Th[k];
        double val = 0.;
        for (auto& q : bary) {
            const R2 P = q[0] * K[0] + q[1] * K[1] + q[2] * K[2];
            val += uh.eval(k, P, 0, op_id) - obstacle_g(P);
        }
        avg[k] = val / 3.;
    }
    return avg;
}


// FunFEM::eval in this archive allocates basis values only up to first
// derivatives (op_dz + 1).  The residual estimator needs dxx and dyy,
// so use a local evaluator that requests and stores second derivatives.
double eval_fun_with_second_derivatives(const Fun& uh, int k, const R2& x, int op) {
    const Space& Vh = uh.getSpace();
    const auto& FK = Vh[k];
    const int ndf = FK.NbDoF();

    std::vector<double> buffer(ndf * Vh.N * (op_All + 1), 0.0);
    RNMK_ w(buffer.data(), ndf, Vh.N, op_All + 1);

    What_d whatd = Fop_D0;
    if (op == op_dx || op == op_dy || op == op_dz ||
        op == op_dxx || op == op_dyy || op == op_dxy) {
        whatd |= Fop_D1;
    }
    if (op == op_dxx || op == op_dyy || op == op_dxy) {
        whatd |= Fop_D2;
    }

    FK.BF(whatd, FK.T.toKref(x), w);

    double val = 0.0;
    for (int j = FK.dfcbegin(0); j < FK.dfcend(0); ++j) {
        val += uh(FK(j)) * w(j, 0, op);
    }
    return val;
}

struct EstimatorParts {
    std::vector<double> total, interior, jump, contact;
    double sum_total = 0., sum_interior = 0., sum_jump = 0., sum_contact = 0.;
    double max_violation = 0.;
    double complementarity = 0.;
};

EstimatorParts compute_estimator(const Mesh& Th, Fun& uh, const std::vector<double>& lambda) {
    static const double bary[4][3] = {
        {1. / 3., 1. / 3., 1. / 3.},
        {0.6, 0.2, 0.2},
        {0.2, 0.6, 0.2},
        {0.2, 0.2, 0.6}
    };
    static const double weight[4] = {-27. / 48., 25. / 48., 25. / 48., 25. / 48.};

    EstimatorParts est;
    est.total.assign(Th.nt, 0.);
    est.interior.assign(Th.nt, 0.);
    est.jump.assign(Th.nt, 0.);
    est.contact.assign(Th.nt, 0.);

    for (int k = Th.first_element(); k < Th.last_element(); k += Th.next_element()) {
        const auto& K = Th[k];
        const double area = K.measure();
        const double hK = K.hMax();
        double eint = 0., econt = 0.;
        for (int iq = 0; iq < 4; ++iq) {
            const R2 P = bary[iq][0] * K[0] + bary[iq][1] * K[1] + bary[iq][2] * K[2];
            const double w = weight[iq] * area;
            const double u = eval_fun_with_second_derivatives(uh, k, P, op_id);
            const double ux = eval_fun_with_second_derivatives(uh, k, P, op_dx);
            const double uy = eval_fun_with_second_derivatives(uh, k, P, op_dy);
            const double lap = eval_fun_with_second_derivatives(uh, k, P, op_dxx)
                             + eval_fun_with_second_derivatives(uh, k, P, op_dyy);
            const double res = lap + lambda[k] + rhs_f(P);
            eint += w * res * res;
            const double gap = obstacle_g(P) - u;
            if (gap > 0.) {
                const double gx = obstacle_gx(P), gy = obstacle_gy(P);
                econt += w * (gap * gap + (gx - ux) * (gx - ux) + (gy - uy) * (gy - uy) + gap * lambda[k]);
                est.max_violation = std::max(est.max_violation, gap);
            }
            est.complementarity += w * std::abs(lambda[k] * (u - obstacle_g(P)));
        }
        est.interior[k] = hK * hK * std::max(0., eint);
        est.contact[k] = std::max(0., econt);
    }

    struct EdgeInfo { int tri = -1; int loc = -1; };
    const int e2v[3][2] = {{1, 2}, {2, 0}, {0, 1}}; // same local convention as Triangle2
    std::map<EdgeKey, EdgeInfo> seen;
    for (int k = Th.first_element(); k < Th.last_element(); k += Th.next_element()) {
        const auto& K = Th[k];
        for (int e = 0; e < 3; ++e) {
            const int a = Th(K[e2v[e][0]]), b = Th(K[e2v[e][1]]);
            const EdgeKey key = edge_key(a, b);
            auto it = seen.find(key);
            if (it == seen.end()) {
                seen[key] = {k, e};
            } else {
                const int k1 = it->second.tri;
                const int e1 = it->second.loc;
                const auto& K1 = Th[k1];
                const auto& K2 = K;
                const R2 A = K[e2v[e][0]];
                const R2 B = K[e2v[e][1]];
                const R2 P(0.5 * (A.x + B.x), 0.5 * (A.y + B.y));
                R2 edge(A, B);
                R2 n(edge.y, -edge.x);
                const double nrm = std::sqrt(n.x * n.x + n.y * n.y);
                n = (1. / nrm) * n;
                const double jx = eval_fun_with_second_derivatives(uh, k1, P, op_dx)
                                - eval_fun_with_second_derivatives(uh, k,  P, op_dx);
                const double jy = eval_fun_with_second_derivatives(uh, k1, P, op_dy)
                                - eval_fun_with_second_derivatives(uh, k,  P, op_dy);
                const double jump = jx * n.x + jy * n.y;
                const double len = R2(A, B).norme();
                const double c1 = 0.25 * K1.hMax() * len * jump * jump;
                const double c2 = 0.25 * K2.hMax() * len * jump * jump;
                est.jump[k1] += c1;
                est.jump[k] += c2;
            }
        }
    }

    for (int k = Th.first_element(); k < Th.last_element(); k += Th.next_element()) {
        est.total[k] = std::max(0., est.interior[k] + est.jump[k] + est.contact[k]);
        est.sum_total += est.total[k];
        est.sum_interior += est.interior[k];
        est.sum_jump += est.jump[k];
        est.sum_contact += est.contact[k];
    }
    return est;
}

MeshResult solve_on_mesh(SimpleMesh& smesh, int level, const Options& opt) {
    const std::string mesh_name = opt.output_prefix + "_mesh_" + std::to_string(level) + ".mesh";
    write_mesh_file(smesh, mesh_name);
    Mesh Th(mesh_name.c_str());

    TypeOfFE_P2BubbleLagrange2d p2b;
    Space Uh(Th, p2b);
    Space Qh(Th, DataFE<Mesh>::P0);

    ProblemOption popt;
    popt.order_space_element_quadrature_ = 7;
    popt.order_space_bord_quadrature_ = 7;
    popt.solver_name_ = opt.solver_name;
    popt.verbose_ = 0;
    popt.clear_matrix_ = false;

    Problem obstacle(Uh, popt);
    obstacle.add(Qh);

    Test u(Uh, 1), v(Uh, 1);
    Test lam(Qh, 1), mu(Qh, 1);
    Fun fh(Uh, fun_f);
    Fun gh(Uh, fun_g);
    Fun zero(Uh, fun_zero);

    obstacle.addBilinear(innerProduct(grad(u), grad(v)) - innerProduct(lam, v) - innerProduct(u, mu), Th);
    obstacle.addLinear(innerProduct(fh.expr(), v) - innerProduct(gh.expr(), mu), Th);
    obstacle.setDirichlet(zero, Th);

    const int nU = Uh.NbDoF();
    const int nQ = Qh.NbDoF();
    const int nTot = nU + nQ;

    Rn current(nTot);
    current = 0.;
    std::vector<bool> active(nQ, false), active_prev(nQ, false);

    double rel = std::numeric_limits<double>::infinity();
    int pd_it = 0;
    int active_count = 0;

    for (pd_it = 0; pd_it < opt.max_pd_iterations; ++pd_it) {
        KN_<double> udata(current(SubArray(nU, 0)));
        Fun uh_prev(Uh, udata);
        std::vector<double> avg = cell_average_u_minus_g(Th, uh_prev);

        active_prev = active;
        active_count = 0;
        for (int k = 0; k < nQ; ++k) {
            const double lambda_k = current(nU + k);
            active[k] = (lambda_k - avg[k] > opt.active_tol);
            if (active[k]) ++active_count;
        }

        std::map<std::pair<int, int>, R> A = obstacle.mat_[0];
        Rn b = obstacle.rhs_;
        for (int k = 0; k < nQ; ++k) {
            if (!active[k]) constrain_dof(A, b, nU + k, 0.);
        }

        Rn previous = current;
        obstacle.solve(A, b);
        current = b;

        double diff2 = 0., norm2 = 0.;
        for (int k = 0; k < nQ; ++k) {
            const double d = current(nU + k) - previous(nU + k);
            diff2 += d * d;
            norm2 += current(nU + k) * current(nU + k);
        }
        rel = std::sqrt(diff2) / std::max(1.0, std::sqrt(norm2));

        int changed = 0;
        for (int k = 0; k < nQ; ++k) if (active[k] != active_prev[k]) ++changed;
        std::cout << "  PDAS it=" << std::setw(2) << pd_it
                  << "  active=" << std::setw(6) << active_count << '/' << nQ
                  << "  changed=" << std::setw(6) << changed
                  << "  rel_dlambda=" << std::scientific << rel << std::defaultfloat << "\n";
        if (rel < opt.pd_tol && changed == 0) break;
    }

    KN_<double> udata(current(SubArray(nU, 0)));
    KN_<double> ldata(current(SubArray(nQ, nU)));
    Fun uh(Uh, udata);
    Fun lambdah(Qh, ldata);

    std::vector<double> lambda(nQ, 0.);
    std::vector<double> active_scalar(nQ, 0.);
    for (int k = 0; k < nQ; ++k) {
        lambda[k] = current(nU + k);
        active_scalar[k] = active[k] ? 1. : 0.;
    }

    EstimatorParts est = compute_estimator(Th, uh, lambda);

    Rn eta_vec(nQ), active_vec(nQ);
    for (int k = 0; k < nQ; ++k) {
        eta_vec(k) = std::sqrt(std::max(0., est.total[k]));
        active_vec(k) = active_scalar[k];
    }
    Fun etah(Qh, eta_vec);
    Fun activeh(Qh, active_vec);

    if (opt.vtk_every > 0 && (level % opt.vtk_every == 0)) {
        const std::string vtk_name = opt.output_prefix + "_" + std::to_string(level) + ".vtk";
        Paraview<Mesh> writer(Th, vtk_name.c_str());
        writer.add(uh, "u", 0, 1);
        writer.add(lambdah, "lambda", 0, 1);
        writer.add(etah, "eta", 0, 1);
        writer.add(activeh, "active", 0, 1);
    }

    MeshResult result;
    result.level = level;
    result.ndof = nTot;
    result.ntri = Th.nt;
    result.active = active_count;
    result.pd_iterations = pd_it + 1;
    result.pd_residual = rel;
    result.eta_total = std::sqrt(std::max(0., est.sum_total));
    result.eta_int = std::sqrt(std::max(0., est.sum_interior));
    result.eta_jump = std::sqrt(std::max(0., est.sum_jump));
    result.eta_contact = std::sqrt(std::max(0., est.sum_contact));
    result.max_violation = est.max_violation;
    result.complementarity = est.complementarity;
    result.eta_cell = est.total;
    result.lambda_cell = lambda;
    result.active_cell = active_scalar;
    return result;
}

std::vector<bool> mark_by_max_strategy(const std::vector<double>& eta2, double beta) {
    std::vector<bool> marked(eta2.size(), false);
    const double max_eta2 = *std::max_element(eta2.begin(), eta2.end());
    const double threshold = beta * beta * max_eta2;
    int nmarked = 0;
    for (std::size_t k = 0; k < eta2.size(); ++k) {
        if (eta2[k] >= threshold && eta2[k] > 0.) {
            marked[k] = true;
            ++nmarked;
        }
    }
    if (nmarked == 0 && !eta2.empty()) {
        marked[std::distance(eta2.begin(), std::max_element(eta2.begin(), eta2.end()))] = true;
    }
    return marked;
}

} // namespace scalar_obstacle

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPIcf mpi(argc, argv);
#else
    (void)argc;
    (void)argv;
#endif

    using namespace scalar_obstacle;

    Options opt;
    // Command line: ./bin/scalar_obstacle [initial_nx] [adaptive_steps]
    if (argc > 1) opt.initial_nx = std::max(3, std::atoi(argv[1]));
    if (argc > 2) opt.adaptive_steps = std::max(0, std::atoi(argv[2]));

    SimpleMesh smesh = make_initial_mesh(opt.initial_nx, opt.initial_nx);

    std::ofstream csv(opt.output_prefix + "_history.csv");
    csv << "level,triangles,dofs,active,pdas_iterations,pdas_residual,eta_total,eta_interior,eta_jump,eta_contact,max_violation,complementarity,rate_vs_prev\n";

    double prev_eta = 0.;
    int prev_ndof = 0;

    std::cout << "Scalar obstacle problem: f=0, g=sin(pi x) sin(pi y)-1/2 on (0,1)^2\n";
    std::cout << "Space: local scalar P2+B for u, P0 for lambda; active-set split by lambda - Pi_0(u-g).\n";
    std::cout << "Outputs: " << opt.output_prefix << "_*.vtk and " << opt.output_prefix << "_history.csv\n\n";

    for (int level = 0; level <= opt.adaptive_steps; ++level) {
        std::cout << "=== adaptive level " << level << " ===\n";
        MeshResult r = solve_on_mesh(smesh, level, opt);

        double rate = 0.;
        if (prev_eta > 0. && prev_ndof > 0 && r.ndof != prev_ndof) {
            rate = std::log(prev_eta / r.eta_total) / std::log(double(r.ndof) / double(prev_ndof));
        }

        std::cout << "level=" << r.level
                  << "  triangles=" << r.ntri
                  << "  dofs=" << r.ndof
                  << "  active=" << r.active
                  << "  eta=" << std::scientific << r.eta_total
                  << "  [int=" << r.eta_int << ", jump=" << r.eta_jump << ", contact=" << r.eta_contact << "]"
                  << "  max(g-u)+=" << r.max_violation
                  << "  comp=" << r.complementarity
                  << "  rate=" << rate << std::defaultfloat << "\n\n";

        csv << r.level << ',' << r.ntri << ',' << r.ndof << ',' << r.active << ','
            << r.pd_iterations << ',' << r.pd_residual << ',' << r.eta_total << ','
            << r.eta_int << ',' << r.eta_jump << ',' << r.eta_contact << ','
            << r.max_violation << ',' << r.complementarity << ',' << rate << "\n";

        if (level == opt.adaptive_steps) break;
        std::vector<bool> marked = mark_by_max_strategy(r.eta_cell, opt.marking_beta);
        const int nmarked = std::accumulate(marked.begin(), marked.end(), 0);
        std::cout << "marking " << nmarked << " / " << marked.size() << " elements with beta=" << opt.marking_beta << "\n\n";
        smesh = refine_marked(smesh, marked);

        prev_eta = r.eta_total;
        prev_ndof = r.ndof;
    }

    std::cout << "History written to " << opt.output_prefix << "_history.csv\n";
    return EXIT_SUCCESS;
}