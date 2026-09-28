#ifndef CUTFEM_OBSTACLE_MAP_BENCHMARKS_HPP
#define CUTFEM_OBSTACLE_MAP_BENCHMARKS_HPP

#include "../../problem/obstacleMap.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cutfem::obstacle::examples {

namespace detail {

inline void require_coordinates(const Vector &q, std::size_t dimension) {
    if (q.size() != dimension)
        throw std::invalid_argument("Obstacle-map chart coordinate dimension mismatch");
}

inline bool finite_feasible_vertices(const std::vector<Vector> &vertices, std::size_t dimension) {
    if (vertices.empty())
        return false;
    for (const auto &q : vertices) {
        if (q.size() != dimension || q.back() < 0.)
            return false;
        for (double value : q)
            if (!std::isfinite(value))
                return false;
    }
    return true;
}

inline void require_tube(double radius) {
    if (!(radius > 0.) || !std::isfinite(radius))
        throw std::invalid_argument("The target tube radius must be finite and positive");
}

inline double tetra_determinant(const SimplexMesh<3> &mesh, const std::array<int, 4> &cell) {
    std::array<Point<3>, 3> edge{};
    for (int j = 0; j < 3; ++j)
        for (int d = 0; d < 3; ++d)
            edge[j][d] = mesh.nodes[cell[j + 1]][d] - mesh.nodes[cell[0]][d];
    return edge[0][0] * (edge[1][1] * edge[2][2] - edge[1][2] * edge[2][1]) -
           edge[0][1] * (edge[1][0] * edge[2][2] - edge[1][2] * edge[2][0]) +
           edge[0][2] * (edge[1][0] * edge[2][1] - edge[1][1] * edge[2][0]);
}

inline void require_mesh_parameters(int subdivisions, double half_width, int dimension) {
    if (subdivisions < 1 || !(half_width > 0.) || !std::isfinite(half_width))
        throw std::invalid_argument("A box mesh needs positive subdivisions and finite positive half-width");
    // Connectivity uses int indices, including in user-supplied meshes.
    double nodes = 1.;
    for (int d = 0; d < dimension; ++d)
        nodes *= static_cast<double>(subdivisions) + 1.;
    if (nodes > std::numeric_limits<int>::max())
        throw std::invalid_argument("The requested box mesh exceeds the node index range");
}

} // namespace detail

// Conforming, fitted P1 meshes. Every cube uses the same body diagonal and
// Freudenthal subdivision, so neighbouring cells have matching face triangles.
inline SimplexMesh<3> box_mesh_3d(int subdivisions = 8, double half_width = 1.) {
    detail::require_mesh_parameters(subdivisions, half_width, 3);
    SimplexMesh<3> mesh;
    const int n = subdivisions;
    const int side = n + 1;
    const auto index = [side](int i, int j, int k) { return (k * side + j) * side + i; };
    mesh.nodes.reserve(static_cast<std::size_t>(side) * side * side);
    mesh.boundary.reserve(static_cast<std::size_t>(side) * side * side);
    mesh.cells.reserve(static_cast<std::size_t>(6) * n * n * n);
    for (int k = 0; k <= n; ++k)
        for (int j = 0; j <= n; ++j)
            for (int i = 0; i <= n; ++i) {
                mesh.nodes.push_back({half_width * (2. * i / n - 1.),
                                      half_width * (2. * j / n - 1.),
                                      half_width * (2. * k / n - 1.)});
                mesh.boundary.push_back(i == 0 || i == n || j == 0 || j == n || k == 0 || k == n);
            }
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                std::array<int, 3> permutation{0, 1, 2};
                do {
                    std::array<int, 3> p{i, j, k};
                    std::array<int, 4> cell{};
                    cell[0] = index(p[0], p[1], p[2]);
                    for (int v = 0; v < 3; ++v) {
                        ++p[permutation[v]];
                        cell[v + 1] = index(p[0], p[1], p[2]);
                    }
                    if (detail::tetra_determinant(mesh, cell) < 0.)
                        std::swap(cell[1], cell[2]);
                    mesh.cells.push_back(cell);
                } while (std::next_permutation(permutation.begin(), permutation.end()));
            }
    return mesh;
}

// A fitted polyhedral approximation of B^3. The continuous radial cube-to-ball
// map sends the cube boundary to the sphere; the FEM domain has planar faces.
// Exact benchmark boundary data are evaluated on this actual polyhedral domain.
inline SimplexMesh<3> unit_ball_mesh_3d(int subdivisions = 8) {
    if (subdivisions < 2)
        throw std::invalid_argument("A ball mesh needs at least two subdivisions per cube edge");
    auto mesh = box_mesh_3d(subdivisions);
    for (auto &p : mesh.nodes) {
        const double norm = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        const double max_norm = std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2])});
        if (norm > 0.)
            for (double &coordinate : p)
                coordinate *= max_norm / norm;
    }
    for (const auto &cell : mesh.cells) {
        const double determinant = detail::tetra_determinant(mesh, cell);
        if (!(determinant > 0.) || !std::isfinite(determinant))
            throw std::runtime_error("Cube-to-ball mesh mapping produced a degenerate or inverted tetrahedron");
    }
    return mesh;
}

inline SimplexMesh<2> box_mesh_2d(int subdivisions = 8, double half_width = 1.) {
    detail::require_mesh_parameters(subdivisions, half_width, 2);
    SimplexMesh<2> mesh;
    const int n = subdivisions;
    const int side = n + 1;
    const auto index = [side](int i, int j) { return j * side + i; };
    mesh.nodes.reserve(static_cast<std::size_t>(side) * side);
    mesh.boundary.reserve(static_cast<std::size_t>(side) * side);
    mesh.cells.reserve(static_cast<std::size_t>(2) * n * n);
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) {
            mesh.nodes.push_back({half_width * (2. * i / n - 1.), half_width * (2. * j / n - 1.)});
            mesh.boundary.push_back(i == 0 || i == n || j == 0 || j == n);
        }
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            mesh.cells.push_back({index(i, j), index(i + 1, j), index(i + 1, j + 1)});
            mesh.cells.push_back({index(i, j), index(i + 1, j + 1), index(i, j + 1)});
        }
    return mesh;
}

// Section 2 of HarmonicObstacleMaps/Main.tex, with the inward Fermi distance
// r = alpha - Phi(y) and an unwrapped longitude s = k*x. The spherical chart
// extends past the collar, but deliberately excludes the north-pole singularity.
struct RotatingProfile {
    double alpha;
    double a;
    double k;

    explicit RotatingProfile(double cap_angle = std::acos(-.5), double contact_half_width = .3,
                             double rotation_rate = 1.)
        : alpha(cap_angle), a(contact_half_width), k(rotation_rate) {
        const double pi = std::acos(-1.);
        if (!(alpha > pi / 2. && alpha < pi) || !(a > 0. && a < 1.) || !std::isfinite(k) ||
            k == 0. || !(std::abs(k) * (1. - a) < pi))
            throw std::invalid_argument("Rotating profile requires pi/2 < alpha < pi, 0 < a < 1, "
                                        "k != 0 and |k|(1-a) < pi");
    }

    // Integrate the regular second-order IVP, avoiding a singular quadrature
    // of its first integral at contact. The returned pair is (Phi(y), Phi'(y)).
    std::pair<double, double> profile(double y) const {
        if (!std::isfinite(y))
            throw std::invalid_argument("Profile argument must be finite");
        const double length = std::abs(y) - a;
        if (length <= 0.)
            return {alpha, 0.};
        if (!(std::abs(k) * length < std::acos(-1.)))
            throw std::invalid_argument("Profile argument leaves the guaranteed decreasing branch");
        const int steps = std::max(1, static_cast<int>(std::ceil(length * 1024. * std::max(1., std::abs(k)))));
        const double step = length / steps;
        const auto acceleration = [rate = k](double phi) { return rate * rate * std::sin(phi) * std::cos(phi); };
        double phi = alpha;
        double velocity = 0.;
        for (int j = 0; j < steps; ++j) {
            const double p1 = velocity;
            const double v1 = acceleration(phi);
            const double p2 = velocity + .5 * step * v1;
            const double v2 = acceleration(phi + .5 * step * p1);
            const double p3 = velocity + .5 * step * v2;
            const double v3 = acceleration(phi + .5 * step * p2);
            const double p4 = velocity + step * v3;
            const double v4 = acceleration(phi + step * p3);
            phi += step * (p1 + 2. * p2 + 2. * p3 + p4) / 6.;
            velocity += step * (v1 + 2. * v2 + 2. * v3 + v4) / 6.;
        }
        return {phi, y < 0. ? -velocity : velocity};
    }

    double phi(double y) const { return profile(y).first; }

    template <std::size_t D> Vector exact(const std::array<double, D> &x) const {
        static_assert(D >= 2, "The rotating profile needs at least two source coordinates");
        return {k * x[0], std::max(0., alpha - phi(x[1]))};
    }

    template <std::size_t D> double reaction(const std::array<double, D> &x) const {
        static_assert(D >= 2, "The rotating profile needs at least two source coordinates");
        return std::abs(x[1]) < a ? -k * k * std::sin(alpha) * std::cos(alpha) : 0.;
    }

    // All free nodes start strictly off the obstacle. No exact contact indicator
    // or exact normal profile is used to initialise their coordinates.
    template <std::size_t D> Vector initial(const std::array<double, D> &x, bool on_boundary) const {
        if (on_boundary)
            return exact(x);
        return {k * x[0] + .04 * std::sin(x[0] + .7 * x[1]),
                .08 + .01 * std::cos(x[0]) + .02 * x[1] * x[1]};
    }

    template <int D> std::vector<Vector> initial(const SimplexMesh<D> &mesh) const {
        if (mesh.boundary.size() != mesh.nodes.size())
            throw std::invalid_argument("Mesh boundary mask must have one entry per node");
        std::vector<Vector> q;
        q.reserve(mesh.nodes.size());
        for (std::size_t i = 0; i < mesh.nodes.size(); ++i)
            q.push_back(initial(mesh.nodes[i], mesh.boundary[i]));
        return q;
    }

    TargetChart chart(double tube_radius = .03) const {
        detail::require_tube(tube_radius);
        if (!(tube_radius < alpha - .05))
            throw std::invalid_argument("The spherical collar must stay below the excluded polar region");
        TargetChart target;
        target.dimension = 2;
        target.tube_radius = tube_radius;
        target.metric = [angle = alpha](const Vector &q, Vector &G, std::vector<Vector> &derivative) {
            detail::require_coordinates(q, 2);
            const double phi = angle - q[1];
            const double sine = std::sin(phi);
            G = {sine * sine, 0., 0., 1.};
            derivative.assign(2, Vector(4, 0.));
            derivative[1][0] = -2. * sine * std::cos(phi);
        };
        target.embed = [angle = alpha](const Vector &q) {
            detail::require_coordinates(q, 2);
            const double phi = angle - q[1];
            return Vector{std::sin(phi) * std::cos(q[0]), std::sin(phi) * std::sin(q[0]), std::cos(phi)};
        };
        // The coordinate image is convex: nodal bounds imply admissibility at
        // every point of every P1 simplex, not just at integration points.
        target.admissible_simplex = [angle = alpha](const std::vector<Vector> &vertices) {
            if (!detail::finite_feasible_vertices(vertices, 2))
                return false;
            for (const auto &q : vertices)
                if (!(q[1] < angle - .05))
                    return false;
            return true;
        };
        return target;
    }
};

// The flat-source counterpart of the exterior-disk warped-product examples in
// mfd-constraint-maps.tex: H'' = k^2 H off contact and H=1, H'=0 at |y|=a.
// This Cartesian cosh profile is a derived verification case, not the paper's
// annular-source Nitsche formula. It solves the full coupled coordinate equations.
struct ExteriorCircleProfile {
    double a;
    double k;

    explicit ExteriorCircleProfile(double contact_half_width = .3, double rotation_rate = 1.)
        : a(contact_half_width), k(rotation_rate) {
        if (!(a > 0. && a < 1.) || !std::isfinite(k) || k == 0.)
            throw std::invalid_argument("Exterior-circle profile requires 0 < a < 1 and finite nonzero k");
    }

    template <std::size_t D> Vector exact(const std::array<double, D> &x) const {
        static_assert(D >= 2, "The exterior-circle profile needs at least two source coordinates");
        const double distance = std::max(std::abs(x[1]) - a, 0.);
        // 2*sinh(z/2)^2 is cosh(z)-1 without cancellation near contact.
        const double sinh_half = std::sinh(.5 * k * distance);
        return {k * x[0], 2. * sinh_half * sinh_half};
    }

    template <std::size_t D> double reaction(const std::array<double, D> &x) const {
        static_assert(D >= 2, "The exterior-circle profile needs at least two source coordinates");
        return std::abs(x[1]) < a ? k * k : 0.;
    }

    template <std::size_t D> Vector initial(const std::array<double, D> &x, bool on_boundary) const {
        if (on_boundary)
            return exact(x);
        return {k * x[0] + .04 * std::sin(x[0] + .7 * x[1]),
                .08 + .01 * std::cos(x[0]) + .02 * x[1] * x[1]};
    }

    template <int D> std::vector<Vector> initial(const SimplexMesh<D> &mesh) const {
        if (mesh.boundary.size() != mesh.nodes.size())
            throw std::invalid_argument("Mesh boundary mask must have one entry per node");
        std::vector<Vector> q;
        q.reserve(mesh.nodes.size());
        for (std::size_t i = 0; i < mesh.nodes.size(); ++i)
            q.push_back(initial(mesh.nodes[i], mesh.boundary[i]));
        return q;
    }

    TargetChart chart(double tube_radius = .05) const {
        detail::require_tube(tube_radius);
        TargetChart target;
        target.dimension = 2;
        target.tube_radius = tube_radius;
        target.metric = [](const Vector &q, Vector &G, std::vector<Vector> &derivative) {
            detail::require_coordinates(q, 2);
            const double radius = 1. + q[1];
            G = {radius * radius, 0., 0., 1.};
            derivative.assign(2, Vector(4, 0.));
            derivative[1][0] = 2. * radius;
        };
        target.embed = [](const Vector &q) {
            detail::require_coordinates(q, 2);
            return Vector{(1. + q[1]) * std::cos(q[0]), (1. + q[1]) * std::sin(q[0])};
        };
        target.admissible_simplex = [](const std::vector<Vector> &vertices) {
            return detail::finite_feasible_vertices(vertices, 2);
        };
        return target;
    }
};

// Arbitrary target dimension, with its final coordinate normal to the boundary.
// Useful both for the scalar obstacle reduction (dimension=1) and flat-map checks.
inline TargetChart euclidean_half_space(int dimension, double tube_radius = .05) {
    if (dimension < 1)
        throw std::invalid_argument("Euclidean half-space dimension must be positive");
    detail::require_tube(tube_radius);
    TargetChart target;
    target.dimension = dimension;
    target.tube_radius = tube_radius;
    target.metric = [dimension](const Vector &q, Vector &G, std::vector<Vector> &derivative) {
        detail::require_coordinates(q, static_cast<std::size_t>(dimension));
        const std::size_t entries = static_cast<std::size_t>(dimension) * dimension;
        G.assign(entries, 0.);
        for (int d = 0; d < dimension; ++d)
            G[static_cast<std::size_t>(d) * dimension + d] = 1.;
        derivative.assign(dimension, Vector(entries, 0.));
    };
    target.embed = [dimension](const Vector &q) {
        detail::require_coordinates(q, static_cast<std::size_t>(dimension));
        return q;
    };
    target.admissible_simplex = [dimension](const std::vector<Vector> &vertices) {
        return detail::finite_feasible_vertices(vertices, static_cast<std::size_t>(dimension));
    };
    return target;
}

// Smooth, globally invertible longitude change theta=s+epsilon*sin(s).
// This exercises tangential metric derivatives without changing target geometry.
inline double inverseLongitude(double theta, double epsilon = .2) {
    if (!std::isfinite(theta) || !std::isfinite(epsilon) || !(std::abs(epsilon) < 1.))
        throw std::invalid_argument("Longitude reparameterization requires finite theta and |epsilon| < 1");
    double lower = theta - std::abs(epsilon);
    double upper = theta + std::abs(epsilon);
    double s = theta;
    for (int iteration = 0; iteration < 64; ++iteration) {
        const double residual = (s - theta) + epsilon * std::sin(s);
        if (std::abs(residual) <= 4. * std::numeric_limits<double>::epsilon() * (1. + std::abs(theta)))
            return s;
        if (residual > 0.)
            upper = s;
        else
            lower = s;
        const double candidate = s - residual / (1. + epsilon * std::cos(s));
        s = candidate > lower && candidate < upper ? candidate : .5 * (lower + upper);
    }
    throw std::runtime_error("Longitude inversion failed to converge");
}

inline TargetChart reparameterizedChart(TargetChart base, double epsilon = .2) {
    if (base.dimension < 2 || !base.metric || !base.embed || !base.admissible_simplex ||
        !std::isfinite(epsilon) || !(std::abs(epsilon) < 1.))
        throw std::invalid_argument("Longitude reparameterization requires a complete chart of dimension >= 2 "
                                    "and |epsilon| < 1");
    detail::require_tube(base.tube_radius);
    TargetChart target;
    target.dimension = base.dimension;
    target.tube_radius = base.tube_radius;
    target.metric = [base, epsilon](const Vector &q, Vector &G, std::vector<Vector> &derivative) {
        const int m = base.dimension;
        detail::require_coordinates(q, static_cast<std::size_t>(m));
        Vector p = q;
        p[0] += epsilon * std::sin(q[0]);
        Vector original;
        std::vector<Vector> original_derivative;
        base.metric(p, original, original_derivative);
        const auto entries = static_cast<std::size_t>(m) * m;
        if (original.size() != entries || original_derivative.size() != static_cast<std::size_t>(m))
            throw std::invalid_argument("Base chart metric has the wrong dimensions");
        for (const auto &dG : original_derivative)
            if (dG.size() != entries)
                throw std::invalid_argument("Base chart metric derivative has the wrong dimensions");
        Vector jacobian(m, 1.);
        jacobian[0] += epsilon * std::cos(q[0]);
        const double second_derivative = -epsilon * std::sin(q[0]);
        G.resize(entries);
        derivative.assign(m, Vector(entries));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) {
                const auto entry = static_cast<std::size_t>(i) * m + j;
                G[entry] = jacobian[i] * jacobian[j] * original[entry];
                for (int c = 0; c < m; ++c) {
                    derivative[c][entry] = jacobian[i] * jacobian[j] * jacobian[c] *
                                           original_derivative[c][entry];
                    if (c == 0) {
                        if (i == 0)
                            derivative[c][entry] += second_derivative * jacobian[j] * original[entry];
                        if (j == 0)
                            derivative[c][entry] += second_derivative * jacobian[i] * original[entry];
                    }
                }
            }
    };
    target.embed = [base, epsilon](const Vector &q) {
        detail::require_coordinates(q, static_cast<std::size_t>(base.dimension));
        Vector p = q;
        p[0] += epsilon * std::sin(q[0]);
        return base.embed(p);
    };
    target.admissible_simplex = [base, epsilon](const std::vector<Vector> &vertices) {
        if (!detail::finite_feasible_vertices(vertices, static_cast<std::size_t>(base.dimension)))
            return false;
        double lower = vertices.front()[0];
        double upper = lower;
        for (const auto &q : vertices) {
            lower = std::min(lower, q[0]);
            upper = std::max(upper, q[0]);
        }
        lower += epsilon * std::sin(lower);
        upper += epsilon * std::sin(upper);
        // The nonlinear image need not lie in the hull of transformed vertices.
        // Certify a containing prism: the transformed longitude interval times
        // the projection of the original simplex onto the other coordinates.
        std::vector<Vector> prism;
        prism.reserve(2 * vertices.size());
        for (const auto &q : vertices) {
            Vector p = q;
            p[0] = lower;
            prism.push_back(p);
            p[0] = upper;
            prism.push_back(std::move(p));
        }
        return base.admissible_simplex(prism);
    };
    return target;
}

} // namespace cutfem::obstacle::examples

#endif // CUTFEM_OBSTACLE_MAP_BENCHMARKS_HPP
