#pragma once

#include "benchmarks.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>

namespace cutfem::obstacle::examples {

struct ExampleOptions {
    int subdivisions = 8;
    bool box = false;
    double tube_radius = .03;
    std::string output_prefix = "obstacle_map";
    ObstacleMapOptions solver;
};

// Integrate the error of F(q_h), not of a linear ambient interpolation of F(q_i).
template <int D> double referenceL2(const ObstacleMapProblem<D>& problem, const Vector& q,
                                   const typename ObstacleMapProblem<D>::Data& reference) {
    const auto& mesh = problem.mesh();
    const int m = problem.target().dimension;
    double error2 = 0.;
    for (const auto& nodes : mesh.cells) {
        const auto cell = cutfem::obstacle::detail::prepare_cell<D>(mesh, nodes);
        for (const auto& phi : cutfem::obstacle::detail::quadrature<D>()) {
            Point<D> x{};
            Vector value(m, 0.);
            for (int j = 0; j <= D; ++j) {
                for (int d = 0; d < D; ++d) x[d] += phi[j]*mesh.nodes[nodes[j]][d];
                for (int a = 0; a < m; ++a) value[a] += phi[j]*q[nodes[j]*m+a];
            }
            const auto actual = problem.target().embed(value);
            const auto exact = problem.target().embed(reference(x));
            if (actual.size() != exact.size()) throw std::invalid_argument("Reference embedding size mismatch");
            for (std::size_t a = 0; a < actual.size(); ++a)
                error2 += cell.volume*problem.sourceMetric(x).density/(D+1)*(actual[a]-exact[a])*(actual[a]-exact[a]);
        }
    }
    return std::sqrt(error2);
}

template <int D> void writeResult(const std::string& prefix, const ObstacleMapProblem<D>& problem,
                                  const ObstacleMapResult& result) {
    const auto& mesh = problem.mesh();
    const int m = problem.target().dimension;
    std::ofstream vtk(prefix + ".vtk");
    if (!vtk) throw std::runtime_error("Cannot open VTK output " + prefix);
    vtk << std::setprecision(17) << "# vtk DataFile Version 3.0\nObstacle map: nodal samples of F(q_h)\nASCII\nDATASET UNSTRUCTURED_GRID\n";
    vtk << "POINTS " << mesh.nodes.size() << " double\n";
    for (const auto& x : mesh.nodes) {
        for (int d = 0; d < 3; ++d) vtk << (d < D ? x[d] : 0.) << ' ';
        vtk << '\n';
    }
    vtk << "CELLS " << mesh.cells.size() << ' ' << (D+2)*mesh.cells.size() << '\n';
    for (const auto& cell : mesh.cells) {
        vtk << D+1;
        for (int node : cell) vtk << ' ' << node;
        vtk << '\n';
    }
    vtk << "CELL_TYPES " << mesh.cells.size() << '\n';
    for (std::size_t k = 0; k < mesh.cells.size(); ++k) vtk << (D == 2 ? 5 : 10) << '\n';
    vtk << "POINT_DATA " << mesh.nodes.size() << '\n';
    auto scalar = [&](const std::string& name, auto data) {
        vtk << "SCALARS " << name << " double 1\nLOOKUP_TABLE default\n";
        for (std::size_t i = 0; i < mesh.nodes.size(); ++i) vtk << data(i) << '\n';
    };
    for (int a = 0; a < m; ++a)
        scalar(a == m-1 ? "r" : "s"+std::to_string(a), [&](std::size_t i) { return result.coordinates[i*m+a]; });
    // Dirichlet reactions are not obstacle multipliers. The separate validity
    // flag permits portable VTK output without serializing NaN values.
    scalar("reaction_valid", [&](std::size_t i) { return mesh.boundary[i] ? 0. : 1.; });
    scalar("reaction", [&](std::size_t i) { return mesh.boundary[i] ? 0. : result.reaction[i]; });
    scalar("contact", [&](std::size_t i) { return result.contact[i] ? 1. : 0.; });
    std::vector<Vector> image;
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i)
        image.push_back(problem.target().embed(Vector(result.coordinates.begin()+i*m, result.coordinates.begin()+(i+1)*m)));
    for (std::size_t a = 0; a < image.front().size(); ++a)
        scalar("u"+std::to_string(a), [&](std::size_t i) { return image[i].at(a); });
    vtk << "CELL_DATA " << mesh.cells.size() << "\nSCALARS contact_cell double 1\nLOOKUP_TABLE default\n";
    for (const auto& cell : mesh.cells) {
        bool contact = true;
        for (int node : cell) contact = contact && result.contact[node];
        vtk << (contact ? 1 : 0) << '\n';
    }
    if (!vtk) throw std::runtime_error("Failed to write VTK output");
    std::ofstream history(prefix + "_energy.csv");
    if (!history) throw std::runtime_error("Cannot open energy history output");
    history << std::setprecision(17) << "accepted_block,energy\n";
    for (std::size_t i = 0; i < result.energy_history.size(); ++i) history << i << ',' << result.energy_history[i] << '\n';
}

// Fluent setup language: all iteration, I/O, and geometry helpers live outside
// mainFiles. A user example only declares mesh, geometry, fields, and options.
template <int D> class ObstacleMapExample {
    using Data = typename ObstacleMapProblem<D>::Data;
    std::optional<SimplexMesh<D>> mesh_;
    std::optional<TargetChart> target_;
    Data boundary_, initial_, reference_;
    ObstacleMapOptions options_;
    std::string prefix_ = "obstacle_map";
  public:
    ObstacleMapExample& on(SimplexMesh<D> mesh) { mesh_ = std::move(mesh); return *this; }
    ObstacleMapExample& into(TargetChart target) { target_ = std::move(target); return *this; }
    ObstacleMapExample& withDirichlet(Data data) { boundary_ = std::move(data); return *this; }
    ObstacleMapExample& startingFrom(Data data) { initial_ = std::move(data); return *this; }
    ObstacleMapExample& compareWith(Data data) { reference_ = std::move(data); return *this; }
    ObstacleMapExample& usingSolver(ObstacleMapOptions options) { options_ = options; return *this; }
    ObstacleMapExample& writeTo(std::string prefix) { prefix_ = std::move(prefix); return *this; }
    int solve() const {
        if (!mesh_ || !target_ || !boundary_ || !initial_) throw std::invalid_argument("Incomplete obstacle-map example");
        const ObstacleMapProblem<D> problem(*mesh_, *target_);
        std::cout << "Fitted P1 obstacle map: " << mesh_->nodes.size() << " nodes, " << mesh_->cells.size() << " simplices\n";
        const auto result = problem.solve(boundary_, initial_, options_, [](const ObstacleMapResult& r) {
            if (r.iterations % 10 == 0)
                std::cout << "iteration=" << r.iterations << " energy=" << std::setprecision(12) << r.energy
                          << " tangent=" << r.tangent_residual << " normal=" << r.normal_residual << std::endl;
        });
        writeResult(prefix_, problem, result);
        const double error = reference_ ? referenceL2(problem, result.coordinates, reference_) : std::numeric_limits<double>::quiet_NaN();
        std::ofstream summary(prefix_ + "_summary.csv");
        if (!summary) throw std::runtime_error("Cannot open summary output");
        summary << "status,iterations,nodes,cells,energy,tangent_residual,normal_residual,reference_l2,normal_steps,tangent_steps,harmonic_steps\n"
                << std::setprecision(17) << result.status << ',' << result.iterations << ',' << mesh_->nodes.size() << ','
                << mesh_->cells.size() << ',' << result.energy << ',' << result.tangent_residual << ',' << result.normal_residual << ','
                << error << ',' << result.normal_steps << ',' << result.tangent_steps << ',' << result.harmonic_steps << '\n';
        std::cout << result.status << " after " << result.iterations << " iterations; residual="
                  << result.tangent_residual+result.normal_residual << "; reference L2=" << error << '\n';
        std::cout << "Output: " << prefix_ << ".vtk, " << prefix_ << "_energy.csv, " << prefix_ << "_summary.csv\n";
        return result.converged ? 0 : 1;
    }
};

template <class Example> int runExample(int argc, char** argv, Example example) {
    try {
        ExampleOptions options;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "3D rotating-profile obstacle map on a fitted polyhedral unit ball.\n"
                             "Options: --subdivisions N --box --tube R --tol T --max-iterations N --output PREFIX\n";
                return 0;
            }
            if (key == "--box") { options.box = true; continue; }
            if (i+1 == argc) throw std::invalid_argument("Missing value for " + key);
            const std::string value = argv[++i];
            std::size_t consumed = 0;
            if (key == "--output") { options.output_prefix = value; continue; }
            if (key == "--subdivisions") options.subdivisions = std::stoi(value, &consumed);
            else if (key == "--max-iterations") options.solver.max_iterations = std::stoi(value, &consumed);
            else if (key == "--tube") options.tube_radius = std::stod(value, &consumed);
            else if (key == "--tol") options.solver.tolerance = std::stod(value, &consumed);
            else throw std::invalid_argument("Unknown option " + key);
            if (consumed != value.size()) throw std::invalid_argument("Invalid numeric value for " + key);
        }
        return example(options);
    } catch (const std::exception& e) {
        std::cerr << "obstacle_map: " << e.what() << '\n';
        return 2;
    }
}

} // namespace cutfem::obstacle::examples
