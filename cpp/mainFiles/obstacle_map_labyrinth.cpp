#include "../example/obstacle_map/driver.hpp"
#include "../example/obstacle_map/labyrinth.hpp"
#include <filesystem>
#include <sstream>

int main(int argc, char** argv) {
    using namespace cutfem::obstacle;
    using cutfem::obstacle::Vector;
    using namespace cutfem::obstacle::examples;
    try {
        LabyrinthGeometry geometry;
        ObstacleMapOptions options;
        options.tolerance=1e-6;
        options.max_qp_iterations=20000;
        int n=12, seed=0;
        double tube=.05;
        bool ball=false;
        std::string prefix="labyrinth", warm_start;
        for (int i=1; i<argc; ++i) {
            const std::string key=argv[i];
            if (key=="--help") {
                std::cout << "Intrinsic curved-target obstacle-map exploration.\n"
                    "--subdivisions N --ball --bias B --frequency F --scale S --twist T\n"
                    "--height H --anisotropy A --warp W --seed N --tube R --tol T\n"
                    "--max-iterations N --max-qp-iterations N --output PREFIX --warm-start NODES.csv\n";
                return 0;
            }
            if (key=="--ball") { ball=true; continue; }
            if (++i==argc) throw std::invalid_argument("Missing value for "+key);
            const std::string value=argv[i];
            if (key=="--output") { prefix=value; continue; }
            if (key=="--warm-start") { warm_start=value; continue; }
            std::size_t end=0;
            if (key=="--subdivisions") n=std::stoi(value,&end);
            else if (key=="--seed") seed=std::stoi(value,&end);
            else if (key=="--max-iterations") options.max_iterations=std::stoi(value,&end);
            else if (key=="--max-qp-iterations") options.max_qp_iterations=std::stoi(value,&end);
            else {
                const double v=std::stod(value,&end);
                if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite option "+key);
                if (key=="--bias") geometry.bias=v;
                else if (key=="--frequency") geometry.frequency=v;
                else if (key=="--scale") geometry.scale=v;
                else if (key=="--twist") geometry.twist=v;
                else if (key=="--height") geometry.height=v;
                else if (key=="--anisotropy") geometry.anisotropy=v;
                else if (key=="--warp") geometry.warp=v;
                else if (key=="--tube") tube=v;
                else if (key=="--tol") options.tolerance=v;
                else throw std::invalid_argument("Unknown option "+key);
            }
            if (end!=value.size()) throw std::invalid_argument("Invalid number for "+key);
        }
        if (!(geometry.height>0. && geometry.height<.3) || geometry.anisotropy<0. ||
            std::abs(geometry.warp)>.4 || std::abs(geometry.bias)>3. ||
            geometry.scale<=0. || geometry.frequency<=0. || seed<0)
            throw std::invalid_argument("Invalid experimental geometry parameters");
        const auto parent=std::filesystem::path(prefix).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        const auto mesh=geometry.mesh(n,ball);
        ObstacleMapProblem<3> problem(mesh,geometry.chart(tube),
            [geometry](const Point<3>& x) { return geometry.source(x); });
        const auto boundary=[geometry](const Point<3>& x) { return geometry.boundary(x); };
        ObstacleMapProblem<3>::Data initial=[geometry,seed](const Point<3>& x) {
            auto q=geometry.boundary(x);
            // Strictly positive normal data; no contact geometry is prescribed.
            q[0]+=.12*std::sin((double(seed)+1.)*1.7+x[1]+2*x[2]);
            q[1]+=.10*std::cos((double(seed)+1.)*2.3+x[0]-x[2]);
            q[2]=.10+.025*std::sin(seed+2*x[0]+x[1])*std::cos(x[2]);
            return q;
        };
        if (!warm_start.empty()) {
            if (ball) throw std::invalid_argument("Warm starts currently require the sheared box source");
            std::ifstream input(warm_start);
            if (!input) throw std::invalid_argument("Cannot read warm-start nodes");
            std::string line;
            std::getline(input,line);
            if (line!="x,y,z,s,t,r,reaction,boundary") throw std::invalid_argument("Unknown warm-start format");
            std::vector<std::array<double,6>> rows;
            while (std::getline(input,line)) {
                std::replace(line.begin(),line.end(),',',' ');
                std::istringstream fields(line);
                std::array<double,6> row;
                for (auto& value:row) if (!(fields>>value) || !std::isfinite(value))
                    throw std::invalid_argument("Malformed warm-start row");
                rows.push_back(row);
            }
            const int side=std::lround(std::cbrt(rows.size())), coarse=side-1;
            if (coarse<1 || std::size_t(side)*side*side!=rows.size())
                throw std::invalid_argument("Warm start is not a structured box grid");
            std::vector<Vector> values(rows.size());
            for (const auto& row:rows) {
                const Point<3> u{row[0]-geometry.warp*row[1]*row[2],row[1],row[2]};
                std::array<int,3> index;
                for (int d=0; d<3; ++d) {
                    const double t=.5*coarse*(u[d]+1.);
                    index[d]=std::lround(t);
                    if (index[d]<0 || index[d]>=side || std::abs(t-index[d])>1e-8)
                        throw std::invalid_argument("Warm start uses a different source warp or mesh");
                }
                auto& value=values[(index[0]*side+index[1])*side+index[2]];
                if (!value.empty() || row[5]<0. || row[5]>1.) throw std::invalid_argument("Invalid warm-start coordinates");
                value={row[3],row[4],row[5]};
            }
            initial=[values=std::move(values),side,coarse,warp=geometry.warp](const Point<3>& x) {
                const Point<3> u{x[0]-warp*x[1]*x[2],x[1],x[2]};
                std::array<int,3> index;
                Point<3> fraction;
                for (int d=0; d<3; ++d) {
                    const double t=std::clamp(.5*coarse*(u[d]+1.),0.,double(coarse));
                    index[d]=std::min(coarse-1,int(std::floor(t)));
                    fraction[d]=t-index[d];
                }
                Vector q(3,0.);
                for (int i=0;i<2;++i) for (int j=0;j<2;++j) for (int k=0;k<2;++k) {
                    const double w=(i?fraction[0]:1-fraction[0])*(j?fraction[1]:1-fraction[1])*(k?fraction[2]:1-fraction[2]);
                    const auto& value=values[((index[0]+i)*side+index[1]+j)*side+index[2]+k];
                    for (int a=0;a<3;++a) q[a]+=w*value[a];
                }
                return q;
            };
        }
        std::cout << "Exploration: " << mesh->nv << " nodes, " << mesh->nt << " tetrahedra\n";
        const auto result=problem.solve(boundary,initial,options,[](const ObstacleMapResult& r) {
            if (r.iterations%10==0) std::cout << "iteration=" << r.iterations << " E="
                << std::setprecision(12) << r.energy << " residual="
                << r.tangent_residual+r.normal_residual << std::endl;
        });
        writeResult(prefix,problem,result);
        // Shared-node topology for analysis: the standard VTK duplicates nodes.
        std::ofstream nodes(prefix+"_nodes.csv"), cells(prefix+"_cells.csv"), summary(prefix+"_summary.csv");
        nodes << std::setprecision(17) << "x,y,z,s,t,r,reaction,boundary\n";
        double minr=1., maxr=0., dual=0., complementarity=0., trace_error=0.;
        int active=0;
        for (int i=0; i<mesh->nv; ++i) {
            const auto x=point<3>((*mesh)(i));
            const auto trace=boundary(x);
            for (double v:x) nodes << v << ',';
            for (int a=0; a<3; ++a) nodes << result.coordinates[3*i+a] << ',';
            nodes << result.reaction[i] << ',' << problem.isBoundary(i) << '\n';
            const double r=result.coordinates[3*i+2];
            minr=std::min(minr,r); maxr=std::max(maxr,r);
            if (problem.isBoundary(i)) {
                for (int a=0; a<3; ++a) trace_error=std::max(trace_error,std::abs(result.coordinates[3*i+a]-trace[a]));
            } else {
                dual=std::max(dual,-result.reaction[i]);
                complementarity=std::max(complementarity,std::abs(r*result.reaction[i]));
                active+=result.contact[i];
            }
        }
        cells << "i,j,k,l\n";
        for (int k=0; k<mesh->nt; ++k) {
            const auto cell=problem.space()[k];
            cells << cell(0) << ',' << cell(1) << ',' << cell(2) << ',' << cell(3) << '\n';
        }
        summary << std::setprecision(17)
            << "status,n,ball,seed,bias,frequency,scale,twist,height,anisotropy,warp,tube,tolerance,iterations,nodes,cells,energy,residual,min_r,max_r,dual_violation,complementarity,trace_error,active_nodes,max_iterations,max_qp_iterations,normal_steps,tangent_steps,harmonic_steps,warm_start\n"
            << result.status << ',' << n << ',' << ball << ',' << seed << ',' << geometry.bias << ','
            << geometry.frequency << ',' << geometry.scale << ',' << geometry.twist << ',' << geometry.height << ','
            << geometry.anisotropy << ',' << geometry.warp << ',' << tube << ',' << options.tolerance << ','
            << result.iterations << ',' << mesh->nv << ',' << mesh->nt << ',' << result.energy << ','
            << result.tangent_residual+result.normal_residual << ',' << minr << ',' << maxr << ',' << dual << ','
            << complementarity << ',' << trace_error << ',' << active << ',' << options.max_iterations << ','
            << options.max_qp_iterations << ',' << result.normal_steps << ',' << result.tangent_steps << ','
            << result.harmonic_steps << ',' << std::quoted(warm_start) << '\n';
        if (!nodes || !cells || !summary) throw std::runtime_error("Failed to write experiment data");
        std::cout << result.status << "; E=" << result.energy << "; residual="
            << result.tangent_residual+result.normal_residual << "; active=" << active << '\n';
        return result.converged ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "obstacle_map_labyrinth: " << e.what() << '\n';
        return 2;
    }
}
