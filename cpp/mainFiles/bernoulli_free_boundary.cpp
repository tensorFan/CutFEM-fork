#include <algorithm>
#include <array>
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
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>


// -----------------------------------------------------------------------------
// Bernoulli free boundary identification demo for the CutFEM library.
//
// This file implements the level-set/CutFEM steepest descent algorithm used in
// Burman--He--Larson, Comparison of shape derivatives using CutFEM for the
// ill-posed Bernoulli free boundary problem, and follows the H1-Riesz velocity
// construction in Burman--Elfverson--Hansbo--Larson--Larsson.
//
// Test case: circular inclusion in the unit square.
//   true free boundary: |x-(1/2,1/2)| = 1/4,
//   level-set sign: phi < 0 in the physical domain Omega = square \ disk,
//   exact data: u = 4 r - 1, f = -4/r, g_D = u on the fixed square boundary,
//   g_N = grad(u) . n on the fixed square boundary.
// -----------------------------------------------------------------------------

namespace bernoulli {

using Sparse = std::map<std::pair<int, int>, double>;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kCx = 0.5;
constexpr double kCy = 0.5;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    Vec2 operator+(const Vec2 &o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2 &o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double a) const { return {a * x, a * y}; }
    Vec2 operator/(double a) const { return {x / a, y / a}; }
};

inline Vec2 operator*(double a, const Vec2 &v) { return v * a; }
inline double dot(const Vec2 &a, const Vec2 &b) { return a.x * b.x + a.y * b.y; }
inline double cross(const Vec2 &a, const Vec2 &b) { return a.x * b.y - a.y * b.x; }
inline double norm(const Vec2 &v) { return std::sqrt(dot(v, v)); }

inline double radius(const Vec2 &p) {
    const double dx = p.x - kCx;
    const double dy = p.y - kCy;
    return std::sqrt(dx * dx + dy * dy);
}

inline double exact_u(const Vec2 &p) { return 4.0 * radius(p) - 1.0; }

inline double rhs_f(const Vec2 &p) {
    const double r = std::max(radius(p), 1e-12);
    return -4.0 / r;
}

inline Vec2 grad_f(const Vec2 &p) {
    const double dx = p.x - kCx;
    const double dy = p.y - kCy;
    const double r = std::max(std::sqrt(dx * dx + dy * dy), 1e-12);
    const double fac = 4.0 / (r * r * r);
    return {fac * dx, fac * dy};
}

inline Vec2 grad_exact_u(const Vec2 &p) {
    const double dx = p.x - kCx;
    const double dy = p.y - kCy;
    const double r = std::max(std::sqrt(dx * dx + dy * dy), 1e-12);
    return {4.0 * dx / r, 4.0 * dy / r};
}

inline Vec2 square_outer_normal(const Vec2 &p) {
    constexpr double eps = 1e-12;
    if (std::abs(p.x) < eps) return {-1.0, 0.0};
    if (std::abs(p.x - 1.0) < eps) return {1.0, 0.0};
    if (std::abs(p.y) < eps) return {0.0, -1.0};
    return {0.0, 1.0};
}

inline double gD_fixed(const Vec2 &p) { return exact_u(p); }
inline double gN_fixed(const Vec2 &p) { return dot(grad_exact_u(p), square_outer_normal(p)); }

struct Mesh {
    int nx = 0;
    double h = 0.0;
    std::vector<Vec2> nodes;
    std::vector<std::array<int, 3>> tris;
    std::vector<std::array<int, 2>> boundary_edges;
    std::vector<std::array<int, 2>> interior_edges;
    std::vector<std::array<int, 2>> edge_tris;

    explicit Mesh(int n = 32) { build(n); }

    int node_id(int i, int j) const { return j * (nx + 1) + i; }

    void build(int n) {
        nx = n;
        h = 1.0 / static_cast<double>(nx);
        nodes.clear();
        tris.clear();
        boundary_edges.clear();
        interior_edges.clear();
        edge_tris.clear();
        nodes.reserve((nx + 1) * (nx + 1));
        for (int j = 0; j <= nx; ++j) {
            for (int i = 0; i <= nx; ++i) nodes.push_back({i * h, j * h});
        }
        tris.reserve(2 * nx * nx);
        for (int j = 0; j < nx; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int n00 = node_id(i, j);
                const int n10 = node_id(i + 1, j);
                const int n01 = node_id(i, j + 1);
                const int n11 = node_id(i + 1, j + 1);
                tris.push_back({n00, n10, n11});
                tris.push_back({n00, n11, n01});
            }
        }
        build_edges();
    }

    void build_edges() {
        std::map<std::pair<int, int>, std::vector<int>> edge_to_tri;
        for (int k = 0; k < static_cast<int>(tris.size()); ++k) {
            const auto &T = tris[k];
            for (int e = 0; e < 3; ++e) {
                int a = T[e];
                int b = T[(e + 1) % 3];
                if (a > b) std::swap(a, b);
                edge_to_tri[{a, b}].push_back(k);
            }
        }
        for (const auto &[edge, adj] : edge_to_tri) {
            if (adj.size() == 1) {
                boundary_edges.push_back({edge.first, edge.second});
            } else if (adj.size() == 2) {
                interior_edges.push_back({edge.first, edge.second});
                edge_tris.push_back({adj[0], adj[1]});
            }
        }
    }
};

struct TriGeom {
    double area = 0.0;
    std::array<Vec2, 3> grad{};
};

TriGeom triangle_geometry(const Mesh &mesh, const std::array<int, 3> &T) {
    const Vec2 a = mesh.nodes[T[0]];
    const Vec2 b = mesh.nodes[T[1]];
    const Vec2 c = mesh.nodes[T[2]];
    const double twoA = cross(b - a, c - a);
    if (twoA <= 0.0) throw std::runtime_error("Triangle orientation is not positive");
    TriGeom g;
    g.area = 0.5 * twoA;
    g.grad[0] = Vec2{b.y - c.y, c.x - b.x} / twoA;
    g.grad[1] = Vec2{c.y - a.y, a.x - c.x} / twoA;
    g.grad[2] = Vec2{a.y - b.y, b.x - a.x} / twoA;
    return g;
}

struct ClipVertex {
    Vec2 p;
    std::array<double, 3> lambda{};
    double phi = 0.0;
};

ClipVertex lerp(const ClipVertex &a, const ClipVertex &b, double t) {
    ClipVertex c;
    c.p = (1.0 - t) * a.p + t * b.p;
    c.phi = (1.0 - t) * a.phi + t * b.phi;
    for (int i = 0; i < 3; ++i) c.lambda[i] = (1.0 - t) * a.lambda[i] + t * b.lambda[i];
    return c;
}

std::vector<ClipVertex> clip_negative_triangle(const Mesh &mesh, const std::array<int, 3> &T,
                                                const std::vector<double> &phi) {
    std::vector<ClipVertex> poly(3);
    for (int i = 0; i < 3; ++i) {
        poly[i].p = mesh.nodes[T[i]];
        poly[i].phi = phi[T[i]];
        poly[i].lambda = {0.0, 0.0, 0.0};
        poly[i].lambda[i] = 1.0;
    }
    auto inside = [](const ClipVertex &v) { return v.phi <= 0.0; };
    std::vector<ClipVertex> out;
    for (int pass = 0; pass < 1; ++pass) {
        if (poly.empty()) break;
        out.clear();
        for (std::size_t i = 0; i < poly.size(); ++i) {
            const ClipVertex &S = poly[i];
            const ClipVertex &E = poly[(i + 1) % poly.size()];
            const bool Sin = inside(S);
            const bool Ein = inside(E);
            if (Sin && Ein) {
                out.push_back(E);
            } else if (Sin && !Ein) {
                const double t = S.phi / (S.phi - E.phi);
                out.push_back(lerp(S, E, t));
            } else if (!Sin && Ein) {
                const double t = S.phi / (S.phi - E.phi);
                out.push_back(lerp(S, E, t));
                out.push_back(E);
            }
        }
        poly.swap(out);
    }
    return poly;
}

std::vector<ClipVertex> interface_segment(const Mesh &mesh, const std::array<int, 3> &T,
                                           const std::vector<double> &phi) {
    std::vector<ClipVertex> seg;
    std::array<ClipVertex, 3> v;
    for (int i = 0; i < 3; ++i) {
        v[i].p = mesh.nodes[T[i]];
        v[i].phi = phi[T[i]];
        v[i].lambda = {0.0, 0.0, 0.0};
        v[i].lambda[i] = 1.0;
    }
    for (int e = 0; e < 3; ++e) {
        const ClipVertex &a = v[e];
        const ClipVertex &b = v[(e + 1) % 3];
        if (a.phi == 0.0) seg.push_back(a);
        if (a.phi * b.phi < 0.0) {
            const double t = a.phi / (a.phi - b.phi);
            seg.push_back(lerp(a, b, t));
        }
    }
    // Remove duplicates created by exact nodal cuts.
    std::vector<ClipVertex> unique;
    for (const auto &q : seg) {
        bool seen = false;
        for (const auto &r : unique) {
            if (norm(q.p - r.p) < 1e-12) seen = true;
        }
        if (!seen) unique.push_back(q);
    }
    if (unique.size() > 2) unique.resize(2);
    return unique;
}

inline void add(Sparse &A, int i, int j, double v) {
    if (std::abs(v) > 1e-30) A[{i, j}] += v;
}

void axpy_rhs(std::vector<double> &b, int i, double v) { b[i] += v; }

void apply_zero_dirichlet(Sparse &A, std::vector<double> &b, int dof) {
    for (auto it = A.begin(); it != A.end();) {
        if (it->first.first == dof || it->first.second == dof) it = A.erase(it);
        else ++it;
    }
    A[{dof, dof}] = 1.0;
    b[dof] = 0.0;
}

std::vector<int> active_nodes_from_phi(const Mesh &mesh, const std::vector<double> &phi) {
    std::vector<int> active(mesh.nodes.size(), 0);
    for (const auto &T : mesh.tris) {
        bool has_neg = false, has_pos = false;
        for (int a : T) {
            has_neg = has_neg || (phi[a] <= 0.0);
            has_pos = has_pos || (phi[a] > 0.0);
        }
        if (has_neg || (has_neg && has_pos)) {
            for (int a : T) active[a] = 1;
        }
    }
    return active;
}

bool is_cut_element(const std::array<int, 3> &T, const std::vector<double> &phi) {
    bool has_neg = false, has_pos = false;
    for (int a : T) {
        has_neg = has_neg || (phi[a] <= 0.0);
        has_pos = has_pos || (phi[a] > 0.0);
    }
    return has_neg && has_pos;
}

bool is_active_element(const std::array<int, 3> &T, const std::vector<double> &phi) {
    for (int a : T) {
        if (phi[a] <= 0.0) return true;
    }
    return is_cut_element(T, phi);
}

struct ProblemData {
    double gamma_ghost = 0.1;
    double beta_nitsche = 10.0;
    double gamma_levelset = 1.0;
};

void assemble_cut_operator_and_primal_rhs(const Mesh &mesh, const std::vector<double> &phi, const ProblemData &data,
                                          Sparse &A, std::vector<double> &rhs) {
    const int N = static_cast<int>(mesh.nodes.size());
    rhs.assign(N, 0.0);
    A.clear();

    std::vector<int> active = active_nodes_from_phi(mesh, phi);
    const double h = mesh.h;

    // Volume and free-boundary Nitsche terms.
    for (int k = 0; k < static_cast<int>(mesh.tris.size()); ++k) {
        const auto &T = mesh.tris[k];
        if (!is_active_element(T, phi)) continue;
        const TriGeom geo = triangle_geometry(mesh, T);
        const auto poly = clip_negative_triangle(mesh, T, phi);
        if (poly.size() < 3) continue;

        for (std::size_t s = 1; s + 1 < poly.size(); ++s) {
            const ClipVertex &a = poly[0];
            const ClipVertex &b = poly[s];
            const ClipVertex &c = poly[s + 1];
            const double area = 0.5 * std::abs(cross(b.p - a.p, c.p - a.p));
            if (area <= 1e-18) continue;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    add(A, T[i], T[j], area * dot(geo.grad[i], geo.grad[j]));
                }
            }

            const double ql[3][3] = {{2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0},
                                     {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0},
                                     {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0}};
            for (int q = 0; q < 3; ++q) {
                ClipVertex xq;
                xq.p = ql[q][0] * a.p + ql[q][1] * b.p + ql[q][2] * c.p;
                for (int i = 0; i < 3; ++i)
                    xq.lambda[i] = ql[q][0] * a.lambda[i] + ql[q][1] * b.lambda[i] + ql[q][2] * c.lambda[i];
                const double w = area / 3.0;
                const double fq = rhs_f(xq.p);
                for (int i = 0; i < 3; ++i) axpy_rhs(rhs, T[i], w * fq * xq.lambda[i]);
            }
        }

        const auto seg = interface_segment(mesh, T, phi);
        if (seg.size() == 2) {
            Vec2 grad_phi{0.0, 0.0};
            for (int i = 0; i < 3; ++i) grad_phi = grad_phi + phi[T[i]] * geo.grad[i];
            const double ng = norm(grad_phi);
            if (ng > 1e-14) {
                Vec2 n = grad_phi / ng; // outward normal of phi<0 domain
                const double L = norm(seg[1].p - seg[0].p);
                const double gp[2] = {0.5 - std::sqrt(1.0 / 12.0), 0.5 + std::sqrt(1.0 / 12.0)};
                for (double tq : gp) {
                    ClipVertex xq = lerp(seg[0], seg[1], tq);
                    const double w = L / 2.0;
                    for (int i = 0; i < 3; ++i) {
                        const double dni = dot(geo.grad[i], n);
                        for (int j = 0; j < 3; ++j) {
                            const double dnj = dot(geo.grad[j], n);
                            const double v = -dni * xq.lambda[j] - dnj * xq.lambda[i]
                                             + data.beta_nitsche / h * xq.lambda[i] * xq.lambda[j];
                            add(A, T[i], T[j], w * v);
                        }
                    }
                }
            }
        }
    }

    // Neumann data on the fixed outer square boundary.
    const double gp[2] = {0.5 - std::sqrt(1.0 / 12.0), 0.5 + std::sqrt(1.0 / 12.0)};
    for (const auto &E : mesh.boundary_edges) {
        const Vec2 a = mesh.nodes[E[0]];
        const Vec2 b = mesh.nodes[E[1]];
        const double L = norm(b - a);
        for (double tq : gp) {
            const Vec2 xq = (1.0 - tq) * a + tq * b;
            const double w = L / 2.0;
            const double gn = gN_fixed(xq);
            axpy_rhs(rhs, E[0], w * gn * (1.0 - tq));
            axpy_rhs(rhs, E[1], w * gn * tq);
        }
    }

    // Ghost penalty near the free boundary.
    for (std::size_t e = 0; e < mesh.interior_edges.size(); ++e) {
        const int k0 = mesh.edge_tris[e][0];
        const int k1 = mesh.edge_tris[e][1];
        if (!is_active_element(mesh.tris[k0], phi) || !is_active_element(mesh.tris[k1], phi)) continue;
        if (!is_cut_element(mesh.tris[k0], phi) && !is_cut_element(mesh.tris[k1], phi)) continue;
        const auto &E = mesh.interior_edges[e];
        Vec2 edge = mesh.nodes[E[1]] - mesh.nodes[E[0]];
        const double L = norm(edge);
        if (L <= 0.0) continue;
        Vec2 n{edge.y / L, -edge.x / L};
        const TriGeom g0 = triangle_geometry(mesh, mesh.tris[k0]);
        const TriGeom g1 = triangle_geometry(mesh, mesh.tris[k1]);
        std::vector<std::pair<int, double>> coeff;
        for (int i = 0; i < 3; ++i) coeff.push_back({mesh.tris[k0][i], dot(g0.grad[i], n)});
        for (int i = 0; i < 3; ++i) coeff.push_back({mesh.tris[k1][i], -dot(g1.grad[i], n)});
        for (auto [ii, ci] : coeff)
            for (auto [jj, cj] : coeff) add(A, ii, jj, data.gamma_ghost * h * L * ci * cj);
    }

    // Deactivate nodes completely inside the obstacle and not touched by active cells.
    for (int i = 0; i < N; ++i) {
        if (!active[i]) apply_zero_dirichlet(A, rhs, i);
    }
}

void assemble_adjoint_rhs_fixed_boundary(const Mesh &mesh, const std::vector<double> &u, std::vector<double> &rhs) {
    const int N = static_cast<int>(mesh.nodes.size());
    rhs.assign(N, 0.0);
    const double invh = 1.0 / mesh.h;
    const double gp[2] = {0.5 - std::sqrt(1.0 / 12.0), 0.5 + std::sqrt(1.0 / 12.0)};
    for (const auto &E : mesh.boundary_edges) {
        const Vec2 a = mesh.nodes[E[0]];
        const Vec2 b = mesh.nodes[E[1]];
        const double L = norm(b - a);
        for (double tq : gp) {
            const Vec2 xq = (1.0 - tq) * a + tq * b;
            const double uq = (1.0 - tq) * u[E[0]] + tq * u[E[1]];
            const double res = uq - gD_fixed(xq);
            const double w = L / 2.0 * invh;
            axpy_rhs(rhs, E[0], w * res * (1.0 - tq));
            axpy_rhs(rhs, E[1], w * res * tq);
        }
    }
}

double compute_cost_fixed_boundary(const Mesh &mesh, const std::vector<double> &u) {
    double J = 0.0;
    const double invh = 1.0 / mesh.h;
    const double gp[2] = {0.5 - std::sqrt(1.0 / 12.0), 0.5 + std::sqrt(1.0 / 12.0)};
    for (const auto &E : mesh.boundary_edges) {
        const Vec2 a = mesh.nodes[E[0]];
        const Vec2 b = mesh.nodes[E[1]];
        const double L = norm(b - a);
        for (double tq : gp) {
            const Vec2 xq = (1.0 - tq) * a + tq * b;
            const double uq = (1.0 - tq) * u[E[0]] + tq * u[E[1]];
            const double r = uq - gD_fixed(xq);
            J += 0.5 * invh * (L / 2.0) * r * r;
        }
    }
    return J;
}

std::vector<double> matvec(const Sparse &A, const std::vector<double> &x) {
    std::vector<double> y(x.size(), 0.0);
    for (const auto &[ij, a] : A) y[ij.first] += a * x[ij.second];
    return y;
}

double dot_vec(const std::vector<double> &a, const std::vector<double> &b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

void axpy_vec(std::vector<double> &y, double a, const std::vector<double> &x) {
    for (std::size_t i = 0; i < y.size(); ++i) y[i] += a * x[i];
}

std::vector<double> solve_sparse(const Sparse &A, const std::vector<double> &b, const std::string &name = "bicgstab") {
    (void)name;
    const int n = static_cast<int>(b.size());
    std::vector<double> x(n, 0.0), r = b, rhat = b, p(n, 0.0), v(n, 0.0), s(n, 0.0), t(n, 0.0);
    std::vector<double> phat(n, 0.0), shat(n, 0.0), diag(n, 1.0);
    for (const auto &[ij, a] : A) {
        if (ij.first == ij.second && std::abs(a) > 1e-30) diag[ij.first] = a;
    }
    const double normb = std::sqrt(std::max(dot_vec(b, b), 1e-300));
    const double tol = 1e-10;
    const int maxit = std::max(5000, 20 * n);
    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double res = std::sqrt(dot_vec(r, r)) / normb;
    if (res < tol) return x;
    for (int it = 0; it < maxit; ++it) {
        const double rho = dot_vec(rhat, r);
        if (std::abs(rho) < 1e-300) break;
        const double beta = (rho / rho_old) * (alpha / omega);
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * (p[i] - omega * v[i]);
        for (int i = 0; i < n; ++i) phat[i] = p[i] / diag[i];
        v = matvec(A, phat);
        const double denom = dot_vec(rhat, v);
        if (std::abs(denom) < 1e-300) break;
        alpha = rho / denom;
        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];
        res = std::sqrt(dot_vec(s, s)) / normb;
        if (res < tol) {
            axpy_vec(x, alpha, phat);
            return x;
        }
        for (int i = 0; i < n; ++i) shat[i] = s[i] / diag[i];
        t = matvec(A, shat);
        const double tt = dot_vec(t, t);
        if (tt < 1e-300) break;
        omega = dot_vec(t, s) / tt;
        axpy_vec(x, alpha, phat);
        axpy_vec(x, omega, shat);
        for (int i = 0; i < n; ++i) r[i] = s[i] - omega * t[i];
        res = std::sqrt(dot_vec(r, r)) / normb;
        if (res < tol) return x;
        if (std::abs(omega) < 1e-300) break;
        rho_old = rho;
    }
    std::cerr << "Warning: BiCGSTAB returned with relative residual " << res << "\n";
    return x;
}

void assemble_velocity_system(const Mesh &mesh, const std::vector<double> &phi, const std::vector<double> &u,
                              const std::vector<double> &p, Sparse &B, std::vector<double> &rhs) {
    const int N = static_cast<int>(mesh.nodes.size());
    rhs.assign(2 * N, 0.0);
    B.clear();

    // H1 inner-product matrix on the background square.
    for (const auto &T : mesh.tris) {
        const TriGeom geo = triangle_geometry(mesh, T);
        const double area = geo.area;
        for (int c = 0; c < 2; ++c) {
            const int off = c * N;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const double mass = area * ((i == j) ? 1.0 / 6.0 : 1.0 / 12.0);
                    const double stiff = area * dot(geo.grad[i], geo.grad[j]);
                    add(B, off + T[i], off + T[j], stiff + mass);
                }
            }
        }
    }

    // Right hand side = -D_Omega L(theta), with continuous domain representation.
    const double ql[3][3] = {{2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0},
                             {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0},
                             {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0}};

    for (const auto &T : mesh.tris) {
        if (!is_active_element(T, phi)) continue;
        const TriGeom geo = triangle_geometry(mesh, T);
        const auto poly = clip_negative_triangle(mesh, T, phi);
        if (poly.size() < 3) continue;

        Vec2 gu{0.0, 0.0};
        Vec2 gpv{0.0, 0.0};
        for (int i = 0; i < 3; ++i) {
            gu = gu + u[T[i]] * geo.grad[i];
            gpv = gpv + p[T[i]] * geo.grad[i];
        }
        const double gu_dot_gp = dot(gu, gpv);

        for (std::size_t s = 1; s + 1 < poly.size(); ++s) {
            const ClipVertex &a = poly[0];
            const ClipVertex &b = poly[s];
            const ClipVertex &c = poly[s + 1];
            const double area = 0.5 * std::abs(cross(b.p - a.p, c.p - a.p));
            if (area <= 1e-18) continue;
            for (int q = 0; q < 3; ++q) {
                ClipVertex xq;
                xq.p = ql[q][0] * a.p + ql[q][1] * b.p + ql[q][2] * c.p;
                for (int i = 0; i < 3; ++i)
                    xq.lambda[i] = ql[q][0] * a.lambda[i] + ql[q][1] * b.lambda[i] + ql[q][2] * c.lambda[i];
                const double w = area / 3.0;
                const double fq = rhs_f(xq.p);
                const Vec2 gfq = grad_f(xq.p);
                double pq = 0.0;
                for (int i = 0; i < 3; ++i) pq += xq.lambda[i] * p[T[i]];
                const double scalar = fq * pq - gu_dot_gp;

                for (int i = 0; i < 3; ++i) {
                    const double lam = xq.lambda[i];
                    const Vec2 gtheta = geo.grad[i];
                    for (int comp = 0; comp < 2; ++comp) {
                        const double div_theta = (comp == 0) ? gtheta.x : gtheta.y;
                        double guSgp = 0.0;
                        const double guv[2] = {gu.x, gu.y};
                        const double gpv_arr[2] = {gpv.x, gpv.y};
                        const double gt[2] = {gtheta.x, gtheta.y};
                        for (int aidx = 0; aidx < 2; ++aidx) {
                            for (int bidx = 0; bidx < 2; ++bidx) {
                                const double Sab = ((aidx == comp) ? gt[bidx] : 0.0)
                                                   + ((bidx == comp) ? gt[aidx] : 0.0);
                                guSgp += guv[aidx] * Sab * gpv_arr[bidx];
                            }
                        }
                        const double gradf_dot_theta = ((comp == 0) ? gfq.x : gfq.y) * lam;
                        const double DL = div_theta * scalar + guSgp + gradf_dot_theta * pq;
                        rhs[comp * N + T[i]] += -w * DL;
                    }
                }
            }
        }
    }

    // Strong homogeneous velocity on the fixed outer boundary.
    std::vector<int> is_boundary(N, 0);
    for (const auto &E : mesh.boundary_edges) {
        is_boundary[E[0]] = 1;
        is_boundary[E[1]] = 1;
    }
    for (int i = 0; i < N; ++i) {
        if (is_boundary[i]) {
            apply_zero_dirichlet(B, rhs, i);
            apply_zero_dirichlet(B, rhs, N + i);
        }
    }
}

double quadratic_form(const Sparse &A, const std::vector<double> &x) {
    double val = 0.0;
    for (const auto &[ij, a] : A) val += x[ij.first] * a * x[ij.second];
    return val;
}

void assemble_levelset_matrices(const Mesh &mesh, const std::vector<double> &beta_x, const std::vector<double> &beta_y,
                                const ProblemData &data, Sparse &M, Sparse &Adv, Sparse &Stab) {
    M.clear();
    Adv.clear();
    Stab.clear();
    const double ql[3][3] = {{2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0},
                             {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0},
                             {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0}};
    for (const auto &T : mesh.tris) {
        const TriGeom geo = triangle_geometry(mesh, T);
        const double area = geo.area;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                add(M, T[i], T[j], area * ((i == j) ? 1.0 / 6.0 : 1.0 / 12.0));
            }
        }
        for (int q = 0; q < 3; ++q) {
            Vec2 beta{0.0, 0.0};
            for (int a = 0; a < 3; ++a) {
                beta.x += ql[q][a] * beta_x[T[a]];
                beta.y += ql[q][a] * beta_y[T[a]];
            }
            const double w = area / 3.0;
            for (int i = 0; i < 3; ++i) {
                const double test = ql[q][i];
                for (int j = 0; j < 3; ++j) {
                    add(Adv, T[i], T[j], w * dot(beta, geo.grad[j]) * test);
                }
            }
        }
    }

    for (std::size_t e = 0; e < mesh.interior_edges.size(); ++e) {
        const int k0 = mesh.edge_tris[e][0];
        const int k1 = mesh.edge_tris[e][1];
        const auto &E = mesh.interior_edges[e];
        Vec2 edge = mesh.nodes[E[1]] - mesh.nodes[E[0]];
        const double L = norm(edge);
        if (L <= 0.0) continue;
        Vec2 n{edge.y / L, -edge.x / L};
        const TriGeom g0 = triangle_geometry(mesh, mesh.tris[k0]);
        const TriGeom g1 = triangle_geometry(mesh, mesh.tris[k1]);
        std::vector<std::pair<int, double>> coeff;
        for (int i = 0; i < 3; ++i) coeff.push_back({mesh.tris[k0][i], dot(g0.grad[i], n)});
        for (int i = 0; i < 3; ++i) coeff.push_back({mesh.tris[k1][i], -dot(g1.grad[i], n)});
        for (auto [ii, ci] : coeff)
            for (auto [jj, cj] : coeff) add(Stab, ii, jj, data.gamma_levelset * mesh.h * mesh.h * L * ci * cj);
    }
}

std::vector<double> apply_matrix(const Sparse &A, const std::vector<double> &x) {
    std::vector<double> y(x.size(), 0.0);
    for (const auto &[ij, a] : A) y[ij.first] += a * x[ij.second];
    return y;
}

std::vector<double> crank_nicolson_levelset_step(const Mesh &mesh, const std::vector<double> &phi,
                                                 const std::vector<double> &beta_x,
                                                 const std::vector<double> &beta_y, double dt,
                                                 const ProblemData &data) {
    const int N = static_cast<int>(mesh.nodes.size());
    Sparse M, Adv, Stab, A;
    assemble_levelset_matrices(mesh, beta_x, beta_y, data, M, Adv, Stab);
    std::vector<double> b(N, 0.0);
    for (const auto &[ij, v] : M) {
        add(A, ij.first, ij.second, v / dt);
        b[ij.first] += v / dt * phi[ij.second];
    }
    for (const auto &[ij, v] : Adv) {
        add(A, ij.first, ij.second, 0.5 * v);
        b[ij.first] += -0.5 * v * phi[ij.second];
    }
    for (const auto &[ij, v] : Stab) {
        add(A, ij.first, ij.second, 0.5 * v);
        b[ij.first] += -0.5 * v * phi[ij.second];
    }
    return solve_sparse(A, b);
}

void write_vtk(const Mesh &mesh, const std::string &filename, const std::vector<double> &phi,
               const std::vector<double> &u, const std::vector<double> &p, const std::vector<double> &bx,
               const std::vector<double> &by) {
    std::ofstream out(filename);
    out << "# vtk DataFile Version 3.0\nBernoulli CutFEM free boundary state\nASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";
    out << "POINTS " << mesh.nodes.size() << " double\n";
    for (const auto &x : mesh.nodes) out << x.x << ' ' << x.y << " 0\n";
    out << "CELLS " << mesh.tris.size() << ' ' << 4 * mesh.tris.size() << "\n";
    for (const auto &T : mesh.tris) out << "3 " << T[0] << ' ' << T[1] << ' ' << T[2] << "\n";
    out << "CELL_TYPES " << mesh.tris.size() << "\n";
    for (std::size_t i = 0; i < mesh.tris.size(); ++i) out << "5\n";
    out << "POINT_DATA " << mesh.nodes.size() << "\n";
    auto scalar = [&](const std::string &name, const std::vector<double> &v) {
        out << "SCALARS " << name << " double 1\nLOOKUP_TABLE default\n";
        for (double z : v) out << z << "\n";
    };
    scalar("level_set", phi);
    scalar("u", u);
    scalar("p", p);
    out << "VECTORS beta double\n";
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) out << bx[i] << ' ' << by[i] << " 0\n";
}

double estimated_radius_from_levelset(const Mesh &mesh, const std::vector<double> &phi) {
    double sum = 0.0;
    double len = 0.0;
    for (const auto &T : mesh.tris) {
        const auto seg = interface_segment(mesh, T, phi);
        if (seg.size() != 2) continue;
        const double L = norm(seg[1].p - seg[0].p);
        const Vec2 mid = 0.5 * (seg[0].p + seg[1].p);
        sum += L * radius(mid);
        len += L;
    }
    return (len > 0.0) ? sum / len : std::numeric_limits<double>::quiet_NaN();
}

std::vector<double> initial_levelset(const Mesh &mesh, double r0) {
    std::vector<double> phi(mesh.nodes.size());
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) phi[i] = r0 - radius(mesh.nodes[i]);
    return phi;
}

struct Options {
    int nx = 40;
    int max_iter = 25;
    int transport_steps = 10;
    double tol = 1e-5;
    double r0 = 0.125;
    double learning_rate = 0.5;
    int vtk_every = 5;
    std::string prefix = "bernoulli_circle";
};

Options parse_options(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const std::string &flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + flag);
            return argv[++i];
        };
        if (a == "--nx") opt.nx = std::stoi(need(a));
        else if (a == "--max-it") opt.max_iter = std::stoi(need(a));
        else if (a == "--transport-steps") opt.transport_steps = std::stoi(need(a));
        else if (a == "--tol") opt.tol = std::stod(need(a));
        else if (a == "--r0") opt.r0 = std::stod(need(a));
        else if (a == "--learning-rate") opt.learning_rate = std::stod(need(a));
        else if (a == "--vtk-every") opt.vtk_every = std::stoi(need(a));
        else if (a == "--prefix") opt.prefix = need(a);
        else if (a == "--help") {
            std::cout << "Usage: bernoulli_free_boundary [--nx N] [--max-it N] [--tol T]\n"
                         "       [--r0 R] [--learning-rate R] [--transport-steps N]\n"
                         "       [--vtk-every N] [--prefix NAME]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option " + a);
        }
    }
    return opt;
}

int run(int argc, char **argv) {
    const Options opt = parse_options(argc, argv);
    Mesh mesh(opt.nx);
    ProblemData data;
    std::vector<double> phi = initial_levelset(mesh, opt.r0);

    std::ofstream csv(opt.prefix + "_history.csv");
    csv << "iter,J,logJ,estimated_radius,beta_H1_norm,transport_time\n";

    std::vector<double> u(mesh.nodes.size(), 0.0), p(mesh.nodes.size(), 0.0), bx(mesh.nodes.size(), 0.0), by(mesh.nodes.size(), 0.0);

    for (int iter = 0; iter <= opt.max_iter; ++iter) {
        Sparse A;
        std::vector<double> rhs_primal;
        assemble_cut_operator_and_primal_rhs(mesh, phi, data, A, rhs_primal);
        u = solve_sparse(A, rhs_primal);

        std::vector<double> rhs_adj;
        assemble_adjoint_rhs_fixed_boundary(mesh, u, rhs_adj);
        std::vector<int> active = active_nodes_from_phi(mesh, phi);
        for (int i = 0; i < static_cast<int>(active.size()); ++i) {
            if (!active[i]) rhs_adj[i] = 0.0;
        }
        p = solve_sparse(A, rhs_adj);

        const double J = compute_cost_fixed_boundary(mesh, u);
        const double r_est = estimated_radius_from_levelset(mesh, phi);

        Sparse B;
        std::vector<double> rhs_beta;
        assemble_velocity_system(mesh, phi, u, p, B, rhs_beta);
        std::vector<double> beta = solve_sparse(B, rhs_beta);
        double beta_norm = std::sqrt(std::max(0.0, quadratic_form(B, beta)));
        if (beta_norm < 1e-14) beta_norm = 1.0;
        for (int i = 0; i < static_cast<int>(mesh.nodes.size()); ++i) {
            bx[i] = beta[i] / beta_norm;
            by[i] = beta[mesh.nodes.size() + i] / beta_norm;
        }

        const double T = opt.learning_rate * J / beta_norm;
        csv << iter << ',' << std::setprecision(16) << J << ',' << std::log(std::max(J, 1e-300)) << ','
            << r_est << ',' << beta_norm << ',' << T << '\n';
        std::cout << "iter " << std::setw(3) << iter << "  J=" << std::scientific << J
                  << "  r_est=" << std::fixed << std::setprecision(6) << r_est
                  << "  |beta|_H1=" << std::scientific << beta_norm << "  T=" << T << std::endl;

        if (opt.vtk_every > 0 && (iter % opt.vtk_every == 0 || iter == opt.max_iter || J <= opt.tol)) {
            std::ostringstream name;
            name << opt.prefix << "_" << std::setw(3) << std::setfill('0') << iter << ".vtk";
            write_vtk(mesh, name.str(), phi, u, p, bx, by);
        }
        if (J <= opt.tol || iter == opt.max_iter) break;

        const double dt = T / static_cast<double>(std::max(1, opt.transport_steps));
        for (int step = 0; step < opt.transport_steps; ++step) {
            phi = crank_nicolson_levelset_step(mesh, phi, bx, by, dt, data);
        }
    }

    std::cout << "Wrote " << opt.prefix << "_history.csv and VTK snapshots with prefix " << opt.prefix << "_*.vtk\n";
    return 0;
}

} // namespace bernoulli

int main(int argc, char **argv) {
    try {
        return bernoulli::run(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "bernoulli_free_boundary: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

