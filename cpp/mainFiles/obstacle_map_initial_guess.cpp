#include "../example/obstacle_map/driver.hpp"

#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>

// A separate experiment: preserve both verification profiles, their original
// initial fields, the example executable, and every nonlinear solver setting.
namespace {
using namespace cutfem::obstacle;
using cutfem::obstacle::Vector;
namespace ex = cutfem::obstacle::examples;

struct Guess {
    std::string name;
    std::uint64_t seed = 0;
};

std::uint64_t mix(std::uint64_t value) {
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

template <std::size_t D, class Profile>
Vector initialValue(const Point<D>& x, const Profile& profile, const Guess& guess, double limit) {
    if (guess.name == "current") return profile.initial(x, false);
    if (guess.name == "all_contact") return {-profile.k*x[0], 0.};
    if (guess.name == "far_normal") return {-profile.k*x[0], limit};
    const double pi = std::acos(-1.);
    if (guess.name == "oscillatory" || guess.name == "angular_only") {
        double phase = 5.3*x[0] + 3.7*x[1];
        if constexpr (D == 3) phase += 2.1*x[2];
        return {-profile.k*x[0] + 4.*pi*std::sin(phase),
                guess.name == "angular_only" ? profile.initial(x, false)[1]
                                            : limit*(.5 + .49*std::cos(1.7*phase + .4))};
    }
    if (guess.name != "random" && guess.name != "normal_only")
        throw std::invalid_argument("Unknown initial guess");
    // Position hashing gives repeatable nodal noise independent of callback
    // order. Canonicalize signed zero, which denotes the same source point.
    auto state = mix(guess.seed + UINT64_C(0x9e3779b97f4a7c15));
    for (double v : x) state = mix(state ^ std::bit_cast<std::uint64_t>(v == 0. ? 0. : v));
    auto uniform = [&]() {
        state = mix(state + UINT64_C(0x9e3779b97f4a7c15));
        return static_cast<double>(state >> 11)*0x1.0p-53;
    };
    const double longitude = 8.*pi*(2.*uniform()-1.);
    return {guess.name == "normal_only" ? profile.k*x[0] : longitude,
            limit*(.001 + .998*uniform())};
}

// Compare the physical maps after P1 coordinate interpolation, using exactly
// the same quadrature as referenceL2 (including its source volume density).
template <int D>
double mapDifference(const ObstacleMapProblem<D>& problem, const Vector& q, const Vector& p) {
    const int m = problem.target().dimension;
    const auto& rule = *QF_Simplex<typename ObstacleMapProblem<D>::Rd>(2);
    std::array<double, D+1> buffer{};
    RNMK_ basis(buffer.data(), D+1, 1, 1);
    double sum = 0.;
    for (int k = 0; k < problem.mesh().nt; ++k) {
        const auto cell = problem.space()[k];
        for (int ip = 0; ip < rule.n; ++ip) {
            const auto& qp = rule[ip];
            cell.BF(Fop_D0, qp, basis);
            Vector qh(m, 0.), ph(m, 0.);
            for (int j = 0; j <= D; ++j) for (int a = 0; a < m; ++a) {
                qh[a] += basis(j,0,op_id)*q[cell(j)*m+a];
                ph[a] += basis(j,0,op_id)*p[cell(j)*m+a];
            }
            const auto u = problem.target().embed(qh), v = problem.target().embed(ph);
            const double weight = cell.T.measure()*qp.a*
                problem.sourceMetric(point<D>(cell.T(qp))).density;
            for (std::size_t a = 0; a < u.size(); ++a) sum += weight*(u[a]-v[a])*(u[a]-v[a]);
        }
    }
    return std::sqrt(sum);
}

template <int D>
void audit(const ObstacleMapProblem<D>& problem, const ObstacleMapResult& result,
           const typename ObstacleMapProblem<D>::Data& trace, const ObstacleMapOptions& options) {
    if (!problem.feasible(result.coordinates)) throw std::runtime_error("Infeasible output map");
    // Reassemble independently: also audit partial results from failed solves.
    const auto assembly = problem.assemble(result.coordinates);
    double tangent = 0., normal = 0.;
    for (int i = 0; i < problem.mesh().nv; ++i) {
        if (problem.isBoundary(i)) {
            const auto value = trace(point<D>(problem.mesh()(i)));
            for (int a = 0; a < 2; ++a)
                if (value[a] != result.coordinates[2*i+a]) throw std::runtime_error("Changed trace");
        } else {
            tangent = std::max(tangent, std::abs(assembly.gradient[2*i]/assembly.mass[i]));
            const double lambda = assembly.gradient[2*i+1]/assembly.mass[i];
            normal = std::max(normal, std::abs(std::min(result.coordinates[2*i+1], lambda)));
            if (std::abs(lambda-result.reaction[i]) > 1e-12) throw std::runtime_error("Stale reaction");
        }
    }
    if (std::abs(tangent-result.tangent_residual) > 1e-12 ||
        std::abs(normal-result.normal_residual) > 1e-12 ||
        (result.converged && tangent+normal > options.tolerance))
        throw std::runtime_error("Incorrect residual or convergence flag");
    for (std::size_t i = 1; i < result.energy_history.size(); ++i)
        if (result.energy_history[i] > result.energy_history[i-1] +
            1e-12*(1.+std::abs(result.energy_history[i-1])))
            throw std::runtime_error("Accepted energy increased");
}

template <int D, class Profile>
int runMesh(const std::string& label, int n, std::shared_ptr<SourceMesh<D>> mesh,
            const Profile& profile, double limit, const std::filesystem::path& output,
            const ObstacleMapOptions& options, std::ofstream& summary, std::ofstream& history) {
    const ObstacleMapProblem<D> problem(mesh, profile.chart(.01));
    const auto trace = [profile](const Point<D>& x) { return profile.exact(x); };
    const std::vector<Guess> guesses{{"current"}, {"all_contact"}, {"far_normal"},
                                    {"oscillatory"}, {"angular_only"}, {"normal_only", 1},
                                    {"random", 1}, {"random", 2}, {"random", 3}};
    ObstacleMapResult baseline;
    int failures = 0;
    for (const auto& guess : guesses) {
        const std::string name = label + "_n" + std::to_string(n) + "_" + guess.name +
            (guess.name == "random" ? "_seed" + std::to_string(guess.seed) : "");
        const auto prefix = (output/name).string();
        const auto initial = [=](const Point<D>& x) { return initialValue<D>(x, profile, guess, limit); };
        auto zero_steps = options;
        zero_steps.max_iterations = 0;
        const auto start = problem.solve(trace, initial, zero_steps);
        ex::writeResult(prefix+"_initial", problem, start);
        const auto write_history = [&](const ObstacleMapResult& r, const char* event) {
            history << label << ',' << n << ',' << guess.name << ',' << guess.seed << ','
                    << r.iterations << ',' << r.energy << ',' << r.tangent_residual << ','
                    << r.normal_residual << ',' << event << '\n';
        };
        std::cout << name << " initial energy=" << start.energy
                  << " residual=" << start.tangent_residual+start.normal_residual << std::endl;
        const auto begin = std::chrono::steady_clock::now();
        const auto result = problem.solve(trace, initial, options, [&](const ObstacleMapResult& r) {
            write_history(r, "outer");
            if (r.iterations % 50 == 0 && r.iterations)
                std::cout << name << " iteration=" << r.iterations << " residual="
                          << r.tangent_residual+r.normal_residual << std::endl;
        });
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        // A failed block can follow accepted blocks in an unfinished cycle.
        // Preserve that returned state even though no outer monitor fires.
        write_history(result, "returned");
        audit<D>(problem, result, trace, options);
        ex::writeResult(prefix, problem, result);
        if (guess.name == "current") {
            if (!result.converged) throw std::runtime_error("Current-IC control failed");
            baseline = result;
        }
        int contact = 0, initial_contact = 0, changed_contact = 0;
        double min_r = std::numeric_limits<double>::infinity(), max_r = 0.;
        double max_abs_s = 0., final_min_r = std::numeric_limits<double>::infinity();
        double final_max_r = 0., dual = 0., complementarity = 0.;
        for (int i = 0; i < mesh->nv; ++i) if (!problem.isBoundary(i)) {
            initial_contact += start.contact[i];
            contact += result.contact[i];
            changed_contact += result.contact[i] != baseline.contact[i];
            min_r = std::min(min_r, start.coordinates[2*i+1]);
            max_r = std::max(max_r, start.coordinates[2*i+1]);
            max_abs_s = std::max(max_abs_s, std::abs(start.coordinates[2*i]));
            final_min_r = std::min(final_min_r, result.coordinates[2*i+1]);
            final_max_r = std::max(final_max_r, result.coordinates[2*i+1]);
            dual = std::max(dual, -result.reaction[i]);
            complementarity = std::max(complementarity, std::abs(result.coordinates[2*i+1]*result.reaction[i]));
        }
        const double error = ex::referenceL2<D>(problem, result.coordinates, trace);
        const double difference = mapDifference<D>(problem, result.coordinates, baseline.coordinates);
        summary << label << ',' << n << ',' << mesh->nv << ',' << mesh->nt << ',' << guess.name << ',' << guess.seed
                << ',' << result.status << ',' << result.iterations << ',' << options.tolerance << ',' << options.max_iterations
                << ',' << start.energy << ',' << start.tangent_residual+start.normal_residual
                << ',' << ex::referenceL2<D>(problem, start.coordinates, trace)
                << ',' << min_r << ',' << max_r << ',' << max_abs_s << ',' << initial_contact
                << ',' << result.energy << ',' << final_min_r << ',' << final_max_r
                << ',' << result.tangent_residual << ',' << result.normal_residual
                << ',' << result.tangent_residual+result.normal_residual << ',' << error
                << ',' << difference << ',' << result.energy-baseline.energy << ',' << contact << ',' << changed_contact
                << ',' << dual << ',' << complementarity << ',' << result.normal_steps << ',' << result.tangent_steps
                << ',' << result.harmonic_steps << ',' << seconds << '\n';
        summary.flush();
        history.flush();
        if (!summary || !history) throw std::runtime_error("Failed to write experiment data");
        std::cout << name << ' ' << result.status << " iterations=" << result.iterations
                  << " residual=" << result.tangent_residual+result.normal_residual
                  << " L2=" << error << " difference from current=" << difference << std::endl;
        failures += !result.converged;
    }
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path output = "obstacle_initial_guess";
        ObstacleMapOptions options;
        // Only the outer limit and tolerance are configurable. All step and QP
        // settings retain the defaults used by test_obstacle_map.cpp.
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "Initial-guess sensitivity: k=1, a=.3, alpha=acos(-.5), tube=.01.\n"
                             "Spherical balls n=4,6; exterior-circle squares n=8,16.\n"
                             "Current, all-contact, far-normal, oscillatory, angular-only, normal-only,\n"
                             "and random seeds 1,2,3.\n"
                             "Options: --output DIRECTORY --max-iterations N --tol T\n"
                             "Exit: 0 all converged, 1 recorded nonconvergence, 2 experiment error.\n";
                return 0;
            }
            if (++i == argc) throw std::invalid_argument("Missing value for "+key);
            const std::string value = argv[i];
            if (key == "--output") { output = value; continue; }
            std::size_t end = 0;
            if (key == "--max-iterations") options.max_iterations = std::stoi(value, &end);
            else if (key == "--tol") options.tolerance = std::stod(value, &end);
            else throw std::invalid_argument("Unknown option "+key);
            if (end != value.size()) throw std::invalid_argument("Invalid number for "+key);
        }
        std::filesystem::create_directories(output);
        std::ofstream summary(output/"results.csv"), history(output/"iterations.csv");
        if (!summary || !history) throw std::runtime_error("Cannot open experiment CSV outputs");
        history << std::setprecision(17)
                << "benchmark,subdivisions,initial_guess,seed,iteration,energy,tangent_residual,normal_residual,event\n";
        summary << std::setprecision(17)
                << "benchmark,subdivisions,nodes,cells,initial_guess,seed,status,iterations,tolerance,max_iterations,"
                   "initial_energy,initial_residual,initial_reference_l2,initial_min_r,initial_max_r,initial_max_abs_s,"
                   "initial_free_contact_nodes,energy,final_min_r,final_max_r,tangent_residual,normal_residual,global_residual,reference_l2,"
                   "physical_l2_difference_from_current,energy_difference_from_current,free_contact_nodes,"
                   "contact_disagreements_from_current,max_dual_violation,max_complementarity,"
                   "normal_steps,tangent_steps,harmonic_steps,solve_seconds\n";
        int failures = 0;
        const ex::RotatingProfile sphere; // k=1, matching the user's table.
        for (int n : {4, 6}) failures += runMesh<3>("spherical_cap", n, ex::unit_ball_mesh_3d(n), sphere,
                                                   sphere.alpha-.051, output, options, summary, history);
        const ex::ExteriorCircleProfile circle;
        for (int n : {8, 16}) failures += runMesh<2>("exterior_circle", n, ex::box_mesh_2d(n), circle,
                                                    20., output, options, summary, history);
        std::cout << "Recorded 36 runs; " << failures << " did not converge. Summary: " << output/"results.csv" << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "obstacle_map_initial_guess: " << e.what() << '\n';
        return 2;
    }
}
