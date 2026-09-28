#pragma once

#include "../solver/boundQuadratic.hpp"
#include <array>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace cutfem::obstacle {

template <int D> using Point = std::array<double, D>;

// Fitted, conforming simplices. Boundary data are imposed at the marked nodes.
// Coordinate FE interpolation is performed BEFORE the target embedding.
template <int D> struct SimplexMesh {
    std::vector<Point<D>> nodes;
    std::vector<std::array<int, D + 1>> cells;
    std::vector<bool> boundary;
};

// A boundary-flattening target chart q=(s^1,...,s^(m-1),r), or regular covering
// parametrization for a map admitting one global lift (e.g. unwrapped angle).
// r is inward geodesic distance only in the declared collar. Beyond it the
// full metric may have cross terms and r may be any nonnegative defining
// coordinate. Metric/derivatives and embedding must describe the SAME chart.
// admissible_simplex certifies the entire convex hull, not just quadrature
// points: this is how a geometry provider certifies pointwise feasibility.
// It must also exclude poles, cut loci, and unsupported chart transitions.
struct TargetChart {
    int dimension = 2;
    double tube_radius = 0.05;
    std::function<void(const Vector&, Vector&, std::vector<Vector>&)> metric;
    std::function<Vector(const Vector&)> embed;
    std::function<bool(const std::vector<Vector>&)> admissible_simplex;
};

template <int D> struct SourceMetric {
    double density = 1.; // sqrt(det(g))
    std::array<Point<D>, D> inverse{};
    SourceMetric() { for (int i = 0; i < D; ++i) inverse[i][i] = 1.; }
};

struct ObstacleMapOptions {
    double tolerance = 1e-7;
    double sigma = 0.1; // metric for steps only; never added to physical energy
    double qp_tolerance = 1e-12;
    double armijo = 1e-4;
    double backtrack = 0.5;
    double contact_tolerance = 1e-7;
    int max_iterations = 500;
    int max_qp_iterations = 1000;
    int max_backtracks = 40;
    int bulk_sweeps = 1;
};

struct ObstacleMapResult {
    Vector coordinates;
    Vector reaction; // g_r / lumped mass, at free nodes; NaN at Dirichlet nodes
    std::vector<bool> contact;
    std::vector<double> energy_history;
    bool converged = false;
    std::string status = "NOT_CONVERGED";
    int iterations = 0;
    int normal_steps = 0, tangent_steps = 0, harmonic_steps = 0;
    double energy = 0.;
    double tangent_residual = std::numeric_limits<double>::infinity();
    double normal_residual = std::numeric_limits<double>::infinity();
};

namespace detail {

inline void require_spd(const Vector& a, int n, const char* name) {
    if (int(a.size()) != n * n) throw std::invalid_argument(std::string(name) + " has wrong size");
    Vector l(n * n, 0.);
    double scale = 0.;
    for (double v : a) {
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(name) + " is not finite");
        scale = std::max(scale, std::abs(v));
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) {
        if (std::abs(a[i*n+j] - a[j*n+i]) > 1e-12 * std::max(1., scale))
            throw std::invalid_argument(std::string(name) + " is not symmetric");
        double v = a[i*n+j];
        for (int k = 0; k < j; ++k) v -= l[i*n+k]*l[j*n+k];
        if (i == j) {
            if (!(v > 1e-14 * scale)) throw std::invalid_argument(std::string(name) + " is singular or not positive definite");
            l[i*n+j] = std::sqrt(v);
        } else l[i*n+j] = v / l[j*n+j];
    }
}

template <int D> struct Cell {
    std::array<int, D + 1> node;
    std::array<Point<D>, D + 1> grad{};
    double volume;
};

template <int D> Cell<D> prepare_cell(const SimplexMesh<D>& mesh, const std::array<int, D+1>& node) {
    static_assert(D == 2 || D == 3, "Triangle and tetrahedron sources are supported");
    Cell<D> out;
    out.node = node;
    std::array<std::array<double, 2*D>, D> a{};
    double scale = 0.;
    for (int id : node) if (id < 0 || id >= int(mesh.nodes.size()))
        throw std::invalid_argument("Cell references an invalid node");
    for (int i = 0; i < D; ++i) for (int j = 0; j < D; ++j) {
        a[i][j] = mesh.nodes[node[j+1]][i] - mesh.nodes[node[0]][i];
        scale = std::max(scale, std::abs(a[i][j]));
        a[i][D+j] = (i == j);
    }
    double det = 1.;
    for (int j = 0; j < D; ++j) {
        int p = j;
        for (int i = j+1; i < D; ++i) if (std::abs(a[i][j]) > std::abs(a[p][j])) p = i;
        if (!(std::abs(a[p][j]) > 1e-13 * scale)) throw std::invalid_argument("Degenerate simplex");
        if (p != j) { std::swap(a[p], a[j]); det = -det; }
        const double pivot = a[j][j];
        det *= pivot;
        for (double& v : a[j]) v /= pivot;
        for (int i = 0; i < D; ++i) if (i != j) {
            const double f = a[i][j];
            for (int k = 0; k < 2*D; ++k) a[i][k] -= f * a[j][k];
        }
    }
    out.volume = std::abs(det) / (D == 2 ? 2. : 6.);
    for (int j = 1; j <= D; ++j) for (int i = 0; i < D; ++i) {
        out.grad[j][i] = a[j-1][D+i];
        out.grad[0][i] -= out.grad[j][i];
    }
    return out;
}

// Positive symmetric degree-two simplex rule; weights sum to one. Energy,
// gradient, and step metric use exactly the same quadrature.
template <int D> std::array<std::array<double, D+1>, D+1> quadrature() {
    std::array<std::array<double, D+1>, D+1> q{};
    const double a = D == 2 ? 2./3. : (5.+3.*std::sqrt(5.))/20.;
    const double b = (1.-a)/D;
    for (int i = 0; i <= D; ++i) for (int j = 0; j <= D; ++j) q[i][j] = (i == j ? a : b);
    return q;
}

} // namespace detail

struct MapAssembly {
    double energy = 0.;
    Vector gradient, mass;
    SparseMatrix stiffness; // frozen full metric, symmetric positive semidefinite
    explicit MapAssembly(int nodes, int m) : gradient(nodes*m, 0.), mass(nodes, 0.), stiffness(nodes*m) {}
};

template <int D> class ObstacleMapProblem {
  public:
    using Data = std::function<Vector(const Point<D>&)>;
    using Source = std::function<SourceMetric<D>(const Point<D>&)>;

    ObstacleMapProblem(SimplexMesh<D> mesh, TargetChart target, Source source = {})
        : mesh_(std::move(mesh)), target_(std::move(target)), source_(std::move(source)) {
        if (mesh_.nodes.empty() || mesh_.cells.empty() || mesh_.boundary.size() != mesh_.nodes.size())
            throw std::invalid_argument("Empty mesh or missing boundary mask");
        for (const auto& p : mesh_.nodes) for (double x : p)
            if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite mesh coordinate");
        if (target_.dimension < 1 || !(target_.tube_radius > 0.) || !std::isfinite(target_.tube_radius) ||
            !target_.metric || !target_.embed || !target_.admissible_simplex)
            throw std::invalid_argument("Incomplete target chart or invalid tube radius");
        for (const auto& cell : mesh_.cells) cells_.push_back(detail::prepare_cell<D>(mesh_, cell));
    }

    const SimplexMesh<D>& mesh() const { return mesh_; }
    const TargetChart& target() const { return target_; }
    SourceMetric<D> sourceMetric(const Point<D>& x) const { return source_ ? source_(x) : SourceMetric<D>{}; }

    Vector interpolate(const Data& data) const {
        Vector q;
        for (const auto& p : mesh_.nodes) {
            const auto v = data(p);
            if (int(v.size()) != target_.dimension) throw std::invalid_argument("Map data has wrong target dimension");
            q.insert(q.end(), v.begin(), v.end());
        }
        return q;
    }

    bool feasible(const Vector& q) const {
        const int m = target_.dimension;
        if (q.size() != mesh_.nodes.size()*m) return false;
        for (double v : q) if (!std::isfinite(v)) return false;
        for (int i = 0; i < int(mesh_.nodes.size()); ++i) if (q[i*m+m-1] < 0.) return false;
        for (const auto& cell : cells_) {
            std::vector<Vector> corners;
            for (int i : cell.node) corners.emplace_back(q.begin()+i*m, q.begin()+(i+1)*m);
            if (!target_.admissible_simplex(corners)) return false;
        }
        return true;
    }

    MapAssembly assemble(const Vector& q, bool with_derivatives = true) const {
        if (!feasible(q)) throw std::invalid_argument("Map leaves the certified target chart; provide another chart or a compatible initial map");
        const int m = target_.dimension, n = int(mesh_.nodes.size());
        MapAssembly out(n, m);
        const auto quadrature = detail::quadrature<D>();
        for (const auto& cell : cells_) {
            std::vector<Point<D>> dq(m);
            for (int j = 0; j <= D; ++j) for (int a = 0; a < m; ++a) for (int d = 0; d < D; ++d)
                dq[a][d] += q[cell.node[j]*m+a]*cell.grad[j][d];
            for (const auto& phi : quadrature) {
                Point<D> x{};
                Vector value(m, 0.);
                for (int j = 0; j <= D; ++j) {
                    for (int a = 0; a < m; ++a) value[a] += phi[j]*q[cell.node[j]*m+a];
                    for (int d = 0; d < D; ++d) x[d] += phi[j]*mesh_.nodes[cell.node[j]][d];
                }
                const auto source = sourceMetric(x);
                if (!(source.density > 0.) || !std::isfinite(source.density)) throw std::invalid_argument("Invalid source volume density");
                Vector inverse;
                for (const auto& row : source.inverse) inverse.insert(inverse.end(), row.begin(), row.end());
                detail::require_spd(inverse, D, "Source inverse metric");
                const double w = cell.volume * source.density / (D+1);
                auto inner = [&](const Point<D>& a, const Point<D>& b) {
                    double v = 0.;
                    for (int i = 0; i < D; ++i) for (int j = 0; j < D; ++j) v += a[i]*source.inverse[i][j]*b[j];
                    return v;
                };
                Vector G;
                std::vector<Vector> dG;
                target_.metric(value, G, dG);
                detail::require_spd(G, m, "Target metric");
                if (with_derivatives) {
                    if (int(dG.size()) != m) throw std::invalid_argument("Missing target metric derivatives");
                    for (const auto& derivative : dG) {
                        if (int(derivative.size()) != m*m) throw std::invalid_argument("Wrong metric derivative size");
                        for (double v : derivative) if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite metric derivative");
                    }
                }
                Vector strain(m*m);
                for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) {
                    strain[a*m+b] = inner(dq[a], dq[b]);
                    out.energy += 0.5*w*G[a*m+b]*strain[a*m+b];
                }
                if (!with_derivatives) continue;
                for (int i = 0; i <= D; ++i) {
                    out.mass[cell.node[i]] += w*phi[i];
                    for (int c = 0; c < m; ++c) {
                        double derivative = 0.;
                        for (int b = 0; b < m; ++b) derivative += G[c*m+b]*inner(cell.grad[i], dq[b]);
                        for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                            derivative += 0.5*phi[i]*dG[c][a*m+b]*strain[a*m+b];
                        out.gradient[cell.node[i]*m+c] += w*derivative;
                    }
                    for (int j = 0; j <= D; ++j) {
                        const double stiffness = w*inner(cell.grad[i], cell.grad[j]);
                        for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b)
                            if (G[a*m+b] != 0.) out.stiffness.add(cell.node[i]*m+a, cell.node[j]*m+b, stiffness*G[a*m+b]);
                    }
                }
            }
        }
        if (!std::isfinite(out.energy)) throw std::runtime_error("Nonfinite energy");
        if (with_derivatives) {
            for (double mass : out.mass) if (!(mass > 0.)) throw std::invalid_argument("Mesh has an unused node");
            for (double g : out.gradient) if (!std::isfinite(g)) throw std::runtime_error("Nonfinite energy derivative");
        }
        return out;
    }

    ObstacleMapResult solve(const Data& boundary, const Data& initial,
                           const ObstacleMapOptions& options = {},
                           const std::function<void(const ObstacleMapResult&)>& monitor = {}) const {
        validate_options(options);
        const int m = target_.dimension, n = int(mesh_.nodes.size());
        if (!boundary || !initial) throw std::invalid_argument("Missing boundary or initial map");
        ObstacleMapResult out;
        out.coordinates = interpolate(initial);
        // The prescribed trace may only be defined on the source boundary.
        for (int i = 0; i < n; ++i) if (mesh_.boundary[i]) {
            const Vector trace = boundary(mesh_.nodes[i]);
            if (int(trace.size()) != m) throw std::invalid_argument("Boundary map has wrong target dimension");
            for (int a = 0; a < m; ++a) out.coordinates[i*m+a] = trace[a];
        }
        auto current = assemble(out.coordinates);
        out.energy_history.push_back(current.energy);
        update_result(out, current, options);
        if (monitor) monitor(out);
        for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
            if (out.tangent_residual + out.normal_residual <= options.tolerance) {
                out.converged = true;
                out.status = "CONVERGED";
                return out;
            }
            // Classification selects solver blocks only, never modifies energy
            // or introduces an internal Dirichlet boundary. Overlap elements
            // always contribute to the shared global residual and step metric.
            auto collar = collar_nodes(out.coordinates);
            for (int block = 0; block < 2 + options.bulk_sweeps; ++block) {
                if (block >= 2) collar = collar_nodes(out.coordinates);
                std::vector<int> selected;
                for (int i = 0; i < n; ++i) if (!mesh_.boundary[i]) {
                    if (block == 0 && collar[i]) selected.push_back(i*m+m-1);
                    if (block == 1 && collar[i]) for (int a = 0; a < m-1; ++a) selected.push_back(i*m+a);
                    if (block >= 2 && !collar[i]) for (int a = 0; a < m; ++a) selected.push_back(i*m+a);
                }
                if (selected.empty()) continue;
                const auto step_status = take_step(out.coordinates, current, selected, options);
                if (!step_status.empty()) {
                    out.status = step_status;
                    update_result(out, current, options);
                    return out;
                }
                if (block == 0) ++out.normal_steps;
                else if (block == 1) ++out.tangent_steps;
                else ++out.harmonic_steps;
                current = assemble(out.coordinates);
                out.energy_history.push_back(current.energy);
            }
            out.iterations = iteration + 1;
            update_result(out, current, options); // BOTH equations at SAME map
            if (monitor) monitor(out);
        }
        if (out.tangent_residual + out.normal_residual <= options.tolerance) {
            out.converged = true;
            out.status = "CONVERGED";
        } else out.status = "MAX_ITERATIONS";
        return out;
    }

  private:
    SimplexMesh<D> mesh_;
    TargetChart target_;
    Source source_;
    std::vector<detail::Cell<D>> cells_;

    static void validate_options(const ObstacleMapOptions& o) {
        if (!(o.tolerance > 0.) || !std::isfinite(o.tolerance) || !(o.sigma > 0.) || !std::isfinite(o.sigma) ||
            !(o.qp_tolerance > 0.) || !std::isfinite(o.qp_tolerance) || !(o.armijo > 0. && o.armijo < 1.) ||
            !(o.backtrack > 0. && o.backtrack < 1.) || o.max_iterations < 0 || o.max_qp_iterations < 1 ||
            o.max_backtracks < 1 || o.bulk_sweeps < 1 || !(o.contact_tolerance >= 0.) || !std::isfinite(o.contact_tolerance))
            throw std::invalid_argument("Invalid obstacle map solver options");
    }

    std::vector<bool> collar_nodes(const Vector& q) const {
        const int m = target_.dimension;
        std::vector<bool> collar(mesh_.nodes.size(), false);
        // One element overlap around the preimage of the collar. Its assembly
        // uses the full metric even if an element straddles the Fermi region.
        for (const auto& cell : cells_) {
            bool near = false;
            for (int i : cell.node) near = near || q[i*m+m-1] < target_.tube_radius;
            if (near) for (int i : cell.node) collar[i] = true;
        }
        return collar;
    }

    std::string take_step(Vector& q, const MapAssembly& current, const std::vector<int>& selected,
                          const ObstacleMapOptions& options) const {
        const int m = target_.dimension, n = int(selected.size());
        std::vector<int> local(q.size(), -1);
        for (int i = 0; i < n; ++i) local[selected[i]] = i;
        SparseMatrix B(n);
        Vector gradient(n), lower(n, -std::numeric_limits<double>::infinity());
        for (int i = 0; i < n; ++i) {
            const int global = selected[i];
            gradient[i] = current.gradient[global];
            if (global % m == m-1) lower[i] = -q[global];
            for (const auto& [column, value] : current.stiffness.rows[global])
                if (local[column] >= 0) B.add(i, local[column], value);
            B.add(i, i, options.sigma * current.mass[global/m]);
        }
        // Inner QP residuals use diag(B), while outer residuals use lumped
        // mass. Tighten the inner tolerance on fine or badly scaled meshes so
        // a zero approximate step cannot hide an unresolved outer equation.
        double inner_tolerance = options.qp_tolerance;
        for (int i = 0; i < n; ++i)
            inner_tolerance = std::min(inner_tolerance, .01*options.tolerance*
                std::min(1., current.mass[selected[i]/m] / B.rows[i].at(i)));
        const auto step = solveBoundQP(B, gradient, lower, inner_tolerance, options.max_qp_iterations);
        if (!step.converged) return "QP_NOT_CONVERGED";
        double slope = 0., norm2 = 0.;
        for (int i = 0; i < n; ++i) { slope += gradient[i]*step.x[i]; norm2 += step.x[i]*step.x[i]; }
        if (norm2 == 0.) return {};
        if (!(slope < 0.) || !std::isfinite(slope)) return "NON_DESCENT_DIRECTION";
        double alpha = 1.;
        for (int attempt = 0; attempt < options.max_backtracks; ++attempt) {
            Vector candidate = q;
            for (int i = 0; i < n; ++i) {
                const int global = selected[i];
                candidate[global] += alpha * step.x[i];
                // Only suppress floating point cancellation at an exact bound.
                if (global % m == m-1 && candidate[global] < 0. && candidate[global] > -1e-14)
                    candidate[global] = 0.;
            }
            if (feasible(candidate)) {
                const double energy = assemble(candidate, false).energy;
                // Near stationarity the predicted decrease can lie below the
                // rounding error of the two global sums. A roundoff allowance
                // avoids shrinking a valid step to zero; convergence still
                // requires the independently assembled full KKT residual.
                const double roundoff = 64.*std::numeric_limits<double>::epsilon()*std::max(1., current.energy);
                if (energy <= current.energy + options.armijo * alpha * slope + roundoff) {
                    q = std::move(candidate);
                    return {};
                }
            }
            alpha *= options.backtrack;
        }
        return "LINE_SEARCH_FAILED";
    }

    void update_result(ObstacleMapResult& out, const MapAssembly& current, const ObstacleMapOptions& options) const {
        const int m = target_.dimension, n = int(mesh_.nodes.size());
        out.energy = current.energy;
        out.tangent_residual = out.normal_residual = 0.;
        out.reaction.assign(n, std::numeric_limits<double>::quiet_NaN());
        out.contact.resize(n);
        for (int i = 0; i < n; ++i) {
            const double r = out.coordinates[i*m+m-1];
            out.contact[i] = r <= options.contact_tolerance;
            if (mesh_.boundary[i]) continue;
            const double lambda = current.gradient[i*m+m-1] / current.mass[i];
            out.reaction[i] = lambda;
            // min(r,lambda) equals r-max(0,r-lambda), without cancellation
            // when a large positive distance is paired with a small residual.
            out.normal_residual = std::max(out.normal_residual, std::abs(std::min(r, lambda)));
            for (int a = 0; a < m-1; ++a)
                out.tangent_residual = std::max(out.tangent_residual, std::abs(current.gradient[i*m+a]/current.mass[i]));
        }
    }
};

// Adapter for existing CutFEM Mesh2/Mesh3 objects, instantiated only if used.
// Uses the full mesh in one process; this solver is not MPI distributed.
template <class Mesh> SimplexMesh<Mesh::D> fittedSimplexMesh(const Mesh& mesh) {
    constexpr int D = Mesh::D;
    using Element = typename Mesh::Element;
    using BoundaryElement = typename Mesh::BorderElement;
    static_assert(D == 2 || D == 3, "Only triangle and tetrahedron meshes are supported");
    static_assert(Element::nv == D+1 && BoundaryElement::nv == D,
                  "Subdivide quadrilateral/hexahedral cells before constructing a simplex mesh");
    SimplexMesh<D> out;
    out.nodes.resize(mesh.nv);
    out.boundary.assign(mesh.nv, false);
    for (int i = 0; i < mesh.nv; ++i) for (int d = 0; d < D; ++d) out.nodes[i][d] = mesh(i)[d];
    out.cells.resize(mesh.nt);
    for (int k = 0; k < mesh.nt; ++k) for (int j = 0; j <= D; ++j) out.cells[k][j] = mesh(k,j);
    for (int f = 0; f < mesh.nbe; ++f) for (int j = 0; j < D; ++j) out.boundary[mesh.be(f,j)] = true;
    return out;
}

} // namespace cutfem::obstacle
