#pragma once

#include "benchmarks.hpp"
#include "../../FESpace/paraview.hpp"
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
    using Rd = typename ObstacleMapProblem<D>::Rd;
    const auto& quadrature = *QF_Simplex<Rd>(2);
    std::array<double, D+1> buffer{};
    RNMK_ basis(buffer.data(), D+1, 1, 1);
    for (int k = 0; k < mesh.nt; ++k) {
        const auto cell = problem.space()[k];
        for (int ip = 0; ip < quadrature.n; ++ip) {
            const auto& qp = quadrature[ip];
            cell.BF(Fop_D0, qp, basis);
            const auto x = point<D>(cell.T(qp));
            Vector value(m, 0.);
            for (int j = 0; j <= D; ++j) for (int a = 0; a < m; ++a)
                value[a] += basis(j,0,op_id)*q[cell(j)*m+a];
            const auto actual = problem.target().embed(value);
            const auto exact = problem.target().embed(reference(x));
            if (actual.size() != exact.size()) throw std::invalid_argument("Reference embedding size mismatch");
            for (std::size_t a = 0; a < actual.size(); ++a)
                error2 += cell.T.measure()*problem.sourceMetric(x).density*qp.a*(actual[a]-exact[a])*(actual[a]-exact[a]);
        }
    }
    return std::sqrt(error2);
}

template <int D> void writeResult(const std::string& prefix, const ObstacleMapProblem<D>& problem,
                                  const ObstacleMapResult& result) {
    const auto& mesh = problem.mesh();
    const int m = problem.target().dimension;
    using Mesh = SourceMesh<D>;
    { std::ofstream check(prefix + ".vtk");
      if (!check) throw std::runtime_error("Cannot open VTK output " + prefix); }
    Paraview<Mesh> vtk(mesh, prefix + ".vtk", 17);
    auto scalar = [&](const std::string& name, auto data) {
        Vector values(problem.space().NbDoF());
        for (int i = 0; i < mesh.nv; ++i) values[i] = data(i);
        FunFEM<Mesh> field(problem.space(), values);
        vtk.add(field, name, 0, 1);
    };
    for (int a = 0; a < m; ++a)
        scalar(a == m-1 ? "r" : "s"+std::to_string(a), [&](std::size_t i) { return result.coordinates[i*m+a]; });
    // Dirichlet reactions are not obstacle multipliers. The separate validity
    // flag permits portable VTK output without serializing NaN values.
    scalar("reaction_valid", [&](std::size_t i) { return problem.isBoundary(i) ? 0. : 1.; });
    scalar("reaction", [&](std::size_t i) { return problem.isBoundary(i) ? 0. : result.reaction[i]; });
    scalar("contact", [&](std::size_t i) { return result.contact[i] ? 1. : 0.; });
    std::vector<Vector> image;
    for (std::size_t i = 0; i < mesh.nv; ++i)
        image.push_back(problem.target().embed(Vector(result.coordinates.begin()+i*m, result.coordinates.begin()+(i+1)*m)));
    for (std::size_t a = 0; a < image.front().size(); ++a)
        scalar("u"+std::to_string(a), [&](std::size_t i) { return image[i].at(a); });
    std::ofstream cells(prefix + ".vtk", std::ios::app);
    cells << "CELL_DATA " << mesh.nt << "\nSCALARS contact_cell double 1\nLOOKUP_TABLE default\n";
    for (int k = 0; k < mesh.nt; ++k) {
        bool contact = true;
        for (int j = 0; j <= D; ++j) contact = contact && result.contact[problem.space()[k](j)];
        cells << (contact ? 1 : 0) << '\n';
    }
    if (!cells) throw std::runtime_error("Failed to write VTK output");
    std::ofstream history(prefix + "_energy.csv");
    if (!history) throw std::runtime_error("Cannot open energy history output");
    history << std::setprecision(17) << "accepted_block,energy\n";
    for (std::size_t i = 0; i < result.energy_history.size(); ++i) history << i << ',' << result.energy_history[i] << '\n';
}

// Fluent setup language: all iteration, I/O, and geometry helpers live outside
// mainFiles. A user example only declares mesh, geometry, fields, and options.
template <int D> class ObstacleMapExample {
    using Data = typename ObstacleMapProblem<D>::Data;
    std::shared_ptr<const SourceMesh<D>> mesh_;
    std::optional<TargetChart> target_;
    Data boundary_, initial_, reference_;
    ObstacleMapOptions options_;
    std::string prefix_ = "obstacle_map";
  public:
    ObstacleMapExample& on(std::shared_ptr<const SourceMesh<D>> mesh) { mesh_ = std::move(mesh); return *this; }
    ObstacleMapExample& into(TargetChart target) { target_ = std::move(target); return *this; }
    ObstacleMapExample& withDirichlet(Data data) { boundary_ = std::move(data); return *this; }
    ObstacleMapExample& startingFrom(Data data) { initial_ = std::move(data); return *this; }
    ObstacleMapExample& compareWith(Data data) { reference_ = std::move(data); return *this; }
    ObstacleMapExample& usingSolver(ObstacleMapOptions options) { options_ = options; return *this; }
    ObstacleMapExample& writeTo(std::string prefix) { prefix_ = std::move(prefix); return *this; }
    int solve() const {
        if (!mesh_ || !target_ || !boundary_ || !initial_) throw std::invalid_argument("Incomplete obstacle-map example");
        const ObstacleMapProblem<D> problem(mesh_, *target_);
        std::cout << "Fitted P1 obstacle map: " << mesh_->nv << " nodes, " << mesh_->nt << " simplices\n";
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
                << std::setprecision(17) << result.status << ',' << result.iterations << ',' << mesh_->nv << ','
                << mesh_->nt << ',' << result.energy << ',' << result.tangent_residual << ',' << result.normal_residual << ','
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
