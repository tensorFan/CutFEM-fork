#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace cutfem::obstacle {

using Vector = std::vector<double>;

// Store both triangles of a symmetric matrix. No solver-library dependency is
// needed, so scalar obstacle subproblems can share this implementation.
struct SparseMatrix {
    std::vector<std::map<int, double>> rows;

    explicit SparseMatrix(int n) {
        if (n < 0) throw std::invalid_argument("Negative matrix size");
        rows.resize(static_cast<std::size_t>(n));
    }
    int size() const { return static_cast<int>(rows.size()); }
    void add(int i, int j, double value) {
        if (i < 0 || j < 0 || i >= size() || j >= size())
            throw std::out_of_range("Sparse matrix index");
        rows[i][j] += value;
    }
    Vector multiply(const Vector& x) const {
        if (static_cast<int>(x.size()) != size())
            throw std::invalid_argument("Matrix/vector size mismatch");
        Vector result(x.size(), 0.);
        for (int i = 0; i < size(); ++i)
            for (const auto& [j, value] : rows[i]) {
                if (j < 0 || j >= size()) throw std::out_of_range("Sparse matrix index");
                result[i] += value * x[j];
            }
        return result;
    }
};

struct BoundQPResult {
    Vector x;
    bool converged = false;
    int iterations = 0;
    double residual = std::numeric_limits<double>::infinity();
};

namespace bound_quadratic_detail {

inline Vector validate(const SparseMatrix& matrix, const Vector& gradient,
                       const Vector& lower, double tolerance, int max_iterations) {
    const int n = matrix.size();
    if (static_cast<int>(gradient.size()) != n || static_cast<int>(lower.size()) != n)
        throw std::invalid_argument("Bound QP vector size mismatch");
    if (!(tolerance > 0.) || !std::isfinite(tolerance) || max_iterations < 0)
        throw std::invalid_argument("Invalid bound QP stopping parameters");
    Vector diagonal(n);
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(gradient[i]) || std::isnan(lower[i]) ||
            lower[i] == std::numeric_limits<double>::infinity())
            throw std::invalid_argument("Nonfinite bound QP data");
        const auto diag = matrix.rows[i].find(i);
        if (diag == matrix.rows[i].end() || !(diag->second > 0.) || !std::isfinite(diag->second))
            throw std::invalid_argument("Bound QP requires positive finite diagonal entries");
        diagonal[i] = diag->second;
        for (const auto& [j, value] : matrix.rows[i]) {
            if (j < 0 || j >= n || !std::isfinite(value))
                throw std::invalid_argument("Invalid bound QP matrix entry");
            const auto transposed = matrix.rows[j].find(i);
            const double other = transposed == matrix.rows[j].end() ? 0. : transposed->second;
            if (std::abs(value - other) > 1e-12 * std::max({1., std::abs(value), std::abs(other)}))
                throw std::invalid_argument("Bound QP requires a symmetric matrix");
        }
    }
    return diagonal;
}

inline double dot(const Vector& a, const Vector& b) {
    double result = 0.;
    for (std::size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}

inline Vector gradientAt(const SparseMatrix& matrix, const Vector& linear, const Vector& x) {
    Vector gradient = matrix.multiply(x);
    for (std::size_t i = 0; i < x.size(); ++i) gradient[i] += linear[i];
    return gradient;
}

// Stable evaluation of ||x - max(lower, x - D^{-1}(Bx+g))||_infinity.
// The residual has the units of x and is independent of uniform energy scaling.
inline double residual(const Vector& x, const Vector& gradient,
                       const Vector& lower, const Vector& diagonal) {
    double result = 0.;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double scaled = gradient[i] / diagonal[i];
        const double value = std::isfinite(lower[i]) ? std::min(x[i] - lower[i], scaled) : scaled;
        if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
        result = std::max(result, std::abs(value));
    }
    return result;
}

// Jacobi-preconditioned CG on the free principal submatrix. Solving for an
// increment with the full current gradient includes all nonzero active values
// in the right hand side, without destructive row/column elimination.
inline Vector reducedPCG(const SparseMatrix& matrix, const Vector& rhs,
                         const Vector& diagonal, const std::vector<bool>& active,
                         double tolerance) {
    const int n = matrix.size();
    Vector solution(n, 0.), residual_vector(n, 0.), direction(n, 0.), z(n, 0.);
    for (int i = 0; i < n; ++i) {
        if (active[i]) continue;
        residual_vector[i] = rhs[i];
        z[i] = residual_vector[i] / diagonal[i];
        direction[i] = z[i];
    }
    double rz = dot(residual_vector, z);
    const int limit = std::max(50, std::min(4 * n, 4000));
    for (int iteration = 0; iteration < limit; ++iteration) {
        double norm = 0.;
        for (int i = 0; i < n; ++i)
            if (!active[i]) norm = std::max(norm, std::abs(residual_vector[i] / diagonal[i]));
        if (norm <= tolerance) break;
        Vector product = matrix.multiply(direction);
        for (int i = 0; i < n; ++i) if (active[i]) product[i] = 0.;
        const double curvature = dot(direction, product);
        // An incomplete CG solve still provides a useful direction. The caller
        // checks descent and uses exact coordinate minimization if necessary.
        if (!(curvature > 0.) || !std::isfinite(curvature) || !(rz > 0.)) break;
        const double alpha = rz / curvature;
        for (int i = 0; i < n; ++i) {
            solution[i] += alpha * direction[i];
            residual_vector[i] -= alpha * product[i];
        }
        if ((iteration + 1) % 50 == 0) {
            const Vector exact = matrix.multiply(solution);
            for (int i = 0; i < n; ++i)
                residual_vector[i] = active[i] ? 0. : rhs[i] - exact[i];
        }
        for (int i = 0; i < n; ++i) z[i] = residual_vector[i] / diagonal[i];
        const double next_rz = dot(residual_vector, z);
        const double beta = next_rz / rz;
        for (int i = 0; i < n; ++i) direction[i] = z[i] + beta * direction[i];
        rz = next_rz;
    }
    return solution;
}

inline void coordinateSweep(const SparseMatrix& matrix, const Vector& linear,
                            const Vector& lower, const Vector& diagonal, Vector& x) {
    for (int i = 0; i < matrix.size(); ++i) {
        double gradient = linear[i];
        for (const auto& [j, value] : matrix.rows[i]) gradient += value * x[j];
        x[i] = std::max(lower[i], x[i] - gradient / diagonal[i]);
    }
}

} // namespace bound_quadratic_detail

// Minimize 1/2 x^T B x + gradient^T x subject to x >= lower, for SPD B.
// A lower bound of -infinity denotes an unrestricted variable. Fixed variables
// should be eliminated by the caller; their coupling belongs in gradient.
// Uses a primal feasible active set, reduced PCG, and coordinate minimization
// as a safeguard. No convergence claim is made unless the current KKT residual
// satisfies tolerance. iterations counts outer active-set iterations.
inline BoundQPResult solveBoundQP(const SparseMatrix& matrix, const Vector& gradient,
                                  const Vector& lower, double tolerance = 1e-10,
                                  int max_iterations = 1000) {
    namespace detail = bound_quadratic_detail;
    const Vector diagonal = detail::validate(matrix, gradient, lower, tolerance, max_iterations);
    const int n = matrix.size();
    BoundQPResult result;
    result.x.resize(n);
    for (int i = 0; i < n; ++i) result.x[i] = std::max(0., lower[i]);
    Vector current_gradient = detail::gradientAt(matrix, gradient, result.x);
    std::vector<bool> active(n, false);
    for (int i = 0; i < n; ++i)
        active[i] = std::isfinite(lower[i]) && result.x[i] == lower[i] && current_gradient[i] >= 0.;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        result.residual = detail::residual(result.x, current_gradient, lower, diagonal);
        if (result.residual <= tolerance) { result.converged = true; return result; }
        result.iterations = iteration + 1;

        double free_residual = 0.;
        int release = -1;
        double most_negative = -tolerance;
        for (int i = 0; i < n; ++i) {
            const double scaled = current_gradient[i] / diagonal[i];
            if (!active[i]) free_residual = std::max(free_residual, std::abs(scaled));
            else if (scaled < most_negative) { most_negative = scaled; release = i; }
        }
        // At a minimizer on the current face, release a constraint with a
        // negative multiplier. Releasing one avoids zero-step active-set cycles.
        if (free_residual <= tolerance && release >= 0) active[release] = false;

        Vector rhs(n);
        for (int i = 0; i < n; ++i) rhs[i] = -current_gradient[i];
        const Vector direction = detail::reducedPCG(matrix, rhs, diagonal, active, 0.1 * tolerance);
        const double slope = detail::dot(current_gradient, direction);
        if (!(slope < 0.) || !std::isfinite(slope)) {
            detail::coordinateSweep(matrix, gradient, lower, diagonal, result.x);
            current_gradient = detail::gradientAt(matrix, gradient, result.x);
            for (int i = 0; i < n; ++i)
                active[i] = std::isfinite(lower[i]) && result.x[i] == lower[i] && current_gradient[i] >= 0.;
            continue;
        }

        double step = 1.;
        Vector blocking_step(n, std::numeric_limits<double>::infinity());
        for (int i = 0; i < n; ++i) {
            if (active[i] || !(direction[i] < 0.) || !std::isfinite(lower[i])) continue;
            blocking_step[i] = std::max(0., (result.x[i] - lower[i]) / -direction[i]);
            step = std::min(step, blocking_step[i]);
        }
        const double rounding = 32. * std::numeric_limits<double>::epsilon() * (1. + step);
        for (int i = 0; i < n; ++i) {
            result.x[i] = std::max(lower[i], result.x[i] + step * direction[i]);
            if (blocking_step[i] <= step + rounding) {
                result.x[i] = lower[i];
                active[i] = true;
            }
        }
        current_gradient = detail::gradientAt(matrix, gradient, result.x);
    }
    result.residual = detail::residual(result.x, current_gradient, lower, diagonal);
    result.converged = result.residual <= tolerance;
    return result;
}

inline BoundQPResult solveSPD(const SparseMatrix& matrix, const Vector& rhs,
                              double tolerance = 1e-10, int max_iterations = 1000) {
    Vector gradient(rhs.size());
    for (std::size_t i = 0; i < rhs.size(); ++i) gradient[i] = -rhs[i];
    return solveBoundQP(matrix, gradient,
                        Vector(rhs.size(), -std::numeric_limits<double>::infinity()),
                        tolerance, max_iterations);
}

} // namespace cutfem::obstacle
