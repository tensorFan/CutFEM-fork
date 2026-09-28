#include "../solver/boundQuadratic.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace cutfem::obstacle;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void checkSolution(const SparseMatrix& matrix, const Vector& gradient,
                   const Vector& lower, const Vector& expected,
                   double tolerance = 1e-10, double error_tolerance = 2e-8) {
    const BoundQPResult result = solveBoundQP(matrix, gradient, lower, tolerance);
    require(result.converged, "QP failed to converge: residual " + std::to_string(result.residual));
    const Vector product = matrix.multiply(result.x);
    double complementarity = 0.;
    for (int i = 0; i < matrix.size(); ++i) {
        require(result.x[i] >= lower[i], "Primal infeasibility");
        require(std::abs(result.x[i] - expected[i]) < error_tolerance,
                "Incorrect minimizer at " + std::to_string(i) + ": " + std::to_string(result.x[i]) +
                " expected " + std::to_string(expected[i]));
        const double multiplier = product[i] + gradient[i];
        if (std::isfinite(lower[i])) {
            require(multiplier > -2e-8, "Negative multiplier");
            complementarity = std::max(complementarity, std::abs(multiplier * (result.x[i] - lower[i])));
        } else require(std::abs(multiplier) < 2e-8, "Unrestricted stationarity failure");
    }
    require(complementarity < 2e-8, "Complementarity failure");
}

void nonzeroActiveValues() {
    SparseMatrix matrix(2);
    matrix.add(0, 0, 4.); matrix.add(0, 1, 1.);
    matrix.add(1, 0, 1.); matrix.add(1, 1, 3.);
    // x_0=-1 contributes to the equation 1*x_0+3*x_1=5.
    checkSolution(matrix, {3.5, -5.}, {-1., 0.5}, {-1., 2.});
    // Positive active values contribute to the same off-diagonal coupling.
    checkSolution(matrix, {-4.5, -7.}, {1., 0.}, {1., 2.});
}

void blockingAndReleasing() {
    SparseMatrix matrix(2);
    matrix.add(0, 0, 2.); matrix.add(0, 1, -1.);
    matrix.add(1, 0, -1.); matrix.add(1, 1, 2.);
    checkSolution(matrix, {-1., 2.}, {0., 0.}, {0.5, 0.});
    // The second variable initially has a positive multiplier but must be
    // released after solving the first free equation.
    checkSolution(matrix, {-3., 0.1}, {0., 0.}, {5.9 / 3., 2.8 / 3.});

    SparseMatrix coupled(2);
    coupled.add(0, 0, 2.); coupled.add(0, 1, 1.5);
    coupled.add(1, 0, 1.5); coupled.add(1, 1, 2.);
    // Both initial gradients are negative, but the unconstrained direction
    // would violate the second bound and must take a blocking step of zero.
    checkSolution(coupled, {-2., -0.1}, {0., 0.}, {1., 0.});
}

void generatedKKTProblems() {
    std::mt19937 engine(57119);
    std::uniform_real_distribution<double> random(-1., 1.);
    const double unrestricted = -std::numeric_limits<double>::infinity();
    for (int trial = 0; trial < 40; ++trial) {
        const int n = 4 + trial;
        std::vector<Vector> factor(n, Vector(n));
        for (auto& row : factor) for (double& value : row) value = random(engine);
        SparseMatrix matrix(n);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            double value = i == j ? 1. : 0.;
            for (int k = 0; k < n; ++k) value += factor[k][i] * factor[k][j];
            matrix.add(i, j, value);
        }
        Vector lower(n), exact(n), multipliers(n, 0.);
        for (int i = 0; i < n; ++i) {
            lower[i] = random(engine);
            exact[i] = lower[i];
            if (i % 3 == 0) multipliers[i] = 0.3 + std::abs(random(engine));
            else exact[i] += 0.2 + std::abs(random(engine));
            if (i % 7 == 1) { lower[i] = unrestricted; multipliers[i] = 0.; }
        }
        Vector gradient = matrix.multiply(exact);
        for (int i = 0; i < n; ++i) gradient[i] = multipliers[i] - gradient[i];
        checkSolution(matrix, gradient, lower, exact);
    }
}

void scaledAndUnrestrictedProblems() {
    SparseMatrix matrix(3);
    matrix.add(0, 0, 1e-8); matrix.add(1, 1, 1.); matrix.add(2, 2, 1e8);
    const auto result = solveSPD(matrix, {2e-8, -3., 4e8});
    require(result.converged, "Scaled SPD solve did not converge");
    require(std::abs(result.x[0] - 2.) < 1e-10 && std::abs(result.x[1] + 3.) < 1e-10 &&
            std::abs(result.x[2] - 4.) < 1e-10, "Scaled SPD solution incorrect");

    SparseMatrix tridiagonal(500);
    for (int i = 0; i < 500; ++i) {
        tridiagonal.add(i, i, 2.);
        if (i > 0) { tridiagonal.add(i, i - 1, -1.); tridiagonal.add(i - 1, i, -1.); }
    }
    Vector exact(500);
    for (int i = 0; i < 500; ++i) exact[i] = std::sin(0.01 * (i + 1));
    const auto sparse_result = solveSPD(tridiagonal, tridiagonal.multiply(exact), 1e-11);
    require(sparse_result.converged, "Sparse SPD solve did not converge");
    for (int i = 0; i < 500; ++i)
        require(std::abs(sparse_result.x[i] - exact[i]) < 2e-8, "Sparse SPD solution incorrect");
}

void stoppingAndValidation() {
    SparseMatrix matrix(1); matrix.add(0, 0, 1.);
    const auto unfinished = solveBoundQP(matrix, {-1.}, {0.}, 1e-10, 0);
    require(!unfinished.converged && unfinished.iterations == 0 && unfinished.residual == 1.,
            "Iteration limit gave false convergence");
    const auto stationary = solveBoundQP(matrix, {1.}, {0.}, 1e-10, 0);
    require(stationary.converged && stationary.iterations == 0, "Initial KKT solution rejected");
    const auto final_step = solveBoundQP(matrix, {-1.}, {0.}, 1e-10, 1);
    require(final_step.converged && final_step.iterations == 1, "Final iterate was not checked");
    const auto empty = solveBoundQP(SparseMatrix(0), {}, {});
    require(empty.converged && empty.x.empty(), "Empty reduced problem failed");

    bool rejected = false;
    try { solveBoundQP(matrix, {0.}, {std::numeric_limits<double>::infinity()}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Infinite positive bound was accepted");
    rejected = false;
    SparseMatrix nonsymmetric(2);
    nonsymmetric.add(0, 0, 1.); nonsymmetric.add(1, 1, 1.); nonsymmetric.add(0, 1, 0.5);
    try { solveBoundQP(nonsymmetric, {0., 0.}, {0., 0.}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Nonsymmetric matrix was accepted");
}

} // namespace

int main() {
    try {
        nonzeroActiveValues();
        blockingAndReleasing();
        generatedKKTProblems();
        scaledAndUnrestrictedProblems();
        stoppingAndValidation();
        std::cout << "Bound quadratic tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Bound quadratic test failure: " << error.what() << '\n';
        return 1;
    }
}
