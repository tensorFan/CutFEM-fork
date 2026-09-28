#pragma once

#include "obstacleMap.hpp"

namespace cutfem::obstacle {

// Jet of an embedding into Euclidean ambient space. For target dimension m,
// jacobian[p*m+a] = d_a F_p and hessian[(p*m+a)*m+b] = d_b d_a F_p.
struct EmbeddingJet {
    Vector value, jacobian, hessian;
};

// q.back() must be inward normal distance in 0 <= r <= tube_radius.
// Outside this collar, a regular chart with a nonnegative defining coordinate
// is sufficient. The caller's certificate must cover each whole coordinate
// convex hull, including feasibility, injectivity (or a valid global lift),
// reach and regularity. No reach estimate or chart atlas is inferred here.
inline TargetChart embeddedTargetChart(
    int m, int ambient_dimension, double tube_radius,
    std::function<EmbeddingJet(const Vector&)> jet,
    std::function<bool(const std::vector<Vector>&)> certificate) {
    if (m < 1 || ambient_dimension < m || !(tube_radius > 0.) ||
        !std::isfinite(tube_radius) || !jet || !certificate)
        throw std::invalid_argument("Invalid embedded target dimensions, collar or callbacks");

    auto checked = [m, ambient_dimension, jet = std::move(jet)](const Vector& q) {
        if (q.size() != static_cast<std::size_t>(m))
            throw std::invalid_argument("Embedding coordinates have wrong size");
        for (double x : q) if (!std::isfinite(x))
            throw std::invalid_argument("Nonfinite embedding coordinate");
        auto out = jet(q);
        const auto jacobian_size = static_cast<std::size_t>(ambient_dimension)*m;
        if (out.value.size() != static_cast<std::size_t>(ambient_dimension) ||
            out.jacobian.size() != jacobian_size || out.hessian.size() != jacobian_size*m)
            throw std::invalid_argument("Embedding jet has wrong dimensions");
        for (const auto* values : {&out.value, &out.jacobian, &out.hessian})
            for (double x : *values) if (!std::isfinite(x))
                throw std::invalid_argument("Nonfinite embedding jet");
        for (int p = 0; p < ambient_dimension; ++p)
            for (int a = 0; a < m; ++a) for (int b = 0; b < a; ++b) {
                const double x = out.hessian[(static_cast<std::size_t>(p)*m+a)*m+b];
                const double y = out.hessian[(static_cast<std::size_t>(p)*m+b)*m+a];
                if (std::abs(x-y) > 1e-12*std::max({1., std::abs(x), std::abs(y)}))
                    throw std::invalid_argument("Embedding Hessian is not symmetric");
            }
        return out;
    };

    TargetChart target;
    target.dimension = m;
    target.tube_radius = tube_radius;
    target.embed = [checked](const Vector& q) { return checked(q).value; };
    target.metric = [m, ambient_dimension, checked](const Vector& q, Vector& G,
                                                   std::vector<Vector>& dG) {
        const auto data = checked(q);
        const auto metric_size = static_cast<std::size_t>(m)*m;
        G.assign(metric_size, 0.);
        dG.assign(m, Vector(metric_size, 0.));
        for (int p = 0; p < ambient_dimension; ++p)
            for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) {
                const auto pa = static_cast<std::size_t>(p)*m+a;
                const auto pb = static_cast<std::size_t>(p)*m+b;
                const auto ab = static_cast<std::size_t>(a)*m+b;
                G[ab] += data.jacobian[pa]*data.jacobian[pb];
                for (int c = 0; c < m; ++c)
                    dG[c][ab] += data.hessian[pa*m+c]*data.jacobian[pb] +
                                data.jacobian[pa]*data.hessian[pb*m+c];
            }
        detail::require_spd(G, m, "Embedded target metric");
        for (const auto& derivative : dG) for (double x : derivative)
            if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite embedded metric derivative");
    };
    target.admissible_simplex = [m, certificate = std::move(certificate)](const std::vector<Vector>& q) {
        if (q.empty()) return false;
        for (const auto& vertex : q) {
            if (vertex.size() != static_cast<std::size_t>(m)) return false;
            for (double x : vertex) if (!std::isfinite(x)) return false;
            if (vertex.back() < 0.) return false;
        }
        return certificate(q);
    };
    return target;
}

} // namespace cutfem::obstacle
