#include "../example/obstacle_map/driver.hpp"
#include "../example/obstacle_map/horosphere.hpp"
#include "../example/obstacle_map/horosphere_mesh.hpp"
#include <filesystem>

int main(int argc, char** argv) {
    using namespace cutfem::obstacle;
    using cutfem::obstacle::Vector;
    using namespace cutfem::obstacle::examples;
    try {
        int n=12, seed=0;
        double radius=.5, slope=1./std::sqrt(3.), tube=.2;
        std::string prefix="horosphere";
        ObstacleMapOptions options;
        options.max_qp_iterations=20000;
        options.tolerance=1e-7;
        for (int i=1; i<argc; ++i) {
            const std::string key=argv[i];
            if (key=="--help") {
                std::cout << "Round contact sphere for a map into the exterior of a hyperbolic horoball.\n"
                    "--subdivisions N --radius A --slope C --seed N --tube T --tol T\n"
                    "--max-iterations N --max-qp-iterations N --output PREFIX\n";
                return 0;
            }
            if (++i==argc) throw std::invalid_argument("Missing value for "+key);
            const std::string value=argv[i];
            if (key=="--output") { prefix=value; continue; }
            std::size_t end=0;
            if (key=="--subdivisions") n=std::stoi(value,&end);
            else if (key=="--seed") seed=std::stoi(value,&end);
            else if (key=="--max-iterations") options.max_iterations=std::stoi(value,&end);
            else if (key=="--max-qp-iterations") options.max_qp_iterations=std::stoi(value,&end);
            else {
                const double v=std::stod(value,&end);
                if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite parameter "+key);
                if (key=="--radius") radius=v;
                else if (key=="--slope") slope=v;
                else if (key=="--tube") tube=v;
                else if (key=="--tol") options.tolerance=v;
                else throw std::invalid_argument("Unknown option "+key);
            }
            if (end!=value.size()) throw std::invalid_argument("Invalid number for "+key);
        }
        if (seed<0) throw std::invalid_argument("Seed must be nonnegative");
        if (n>0 && n%2 && n<std::numeric_limits<int>::max()) ++n;
        const HorosphereProfile profile(radius,slope);
        const auto parent=std::filesystem::path(prefix).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        const auto mesh=horosphere_ball_mesh_3d(n);
        ObstacleMapProblem<3> problem(mesh,profile.chart(tube));
        const auto trace=[profile](const Point<3>& x){ return profile.exact(x); };
        const auto initial=[profile,seed](const Point<3>& x){ return profile.initial(x,seed); };
        std::cout << "Horosphere map: " << mesh->nv << " nodes, " << mesh->nt << " tetrahedra\n";
        const auto result=problem.solve(trace,initial,options,[](const ObstacleMapResult& r){
            std::cout << "iteration=" << r.iterations << " energy=" << std::setprecision(12)
                << r.energy << " residual=" << r.tangent_residual+r.normal_residual << std::endl;
        });
        writeResult(prefix,problem,result);
        std::ofstream nodes(prefix+"_nodes.csv"), cells(prefix+"_cells.csv"), reference(prefix+"_profile.csv");
        nodes << std::setprecision(17) << "x,y,z,s0,s1,s2,r,reaction,boundary,r_reference,s0_reference,s1_reference,s2_reference\n";
        double max_error=0., trace_error=0., minr=1., maxr=0., dual=0., complementarity=0.;
        double contact_volume=0., interior_reaction_error=0.;
        int active=0;
        for (int i=0; i<mesh->nv; ++i) {
            const auto x=point<3>((*mesh)(i));
            const auto exact=profile.exact(x);
            for (double v:x) nodes << v << ',';
            for (int a=0; a<4; ++a) {
                nodes << result.coordinates[4*i+a] << ',';
                max_error=std::max(max_error,std::abs(result.coordinates[4*i+a]-exact[a]));
                if (problem.isBoundary(i)) trace_error=std::max(trace_error,std::abs(result.coordinates[4*i+a]-exact[a]));
            }
            nodes << result.reaction[i] << ',' << problem.isBoundary(i) << ',' << exact[3];
            for (int a=0;a<3;++a) nodes << ',' << exact[a];
            nodes << '\n';
            const double r=result.coordinates[4*i+3];
            minr=std::min(minr,r); maxr=std::max(maxr,r);
            if (!problem.isBoundary(i)) {
                dual=std::max(dual,-result.reaction[i]);
                complementarity=std::max(complementarity,std::abs(r*result.reaction[i]));
                active+=result.contact[i];
                const double rho=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
                if (rho<radius-2./n)
                    interior_reaction_error=std::max(interior_reaction_error,std::abs(result.reaction[i]-3*slope*slope));
            }
        }
        cells << "i,j,k,l\n";
        for (int k=0;k<mesh->nt;++k) {
            const auto cell=problem.space()[k];
            cells << cell(0) << ',' << cell(1) << ',' << cell(2) << ',' << cell(3) << '\n';
            bool contact=true;
            for (int j=0;j<4;++j) contact=contact && result.contact[cell(j)];
            if (contact) contact_volume+=cell.T.measure();
        }
        reference << std::setprecision(17) << "rho,f,r,fp,rp\n";
        for (int i=0;i<=2000;++i) {
            const double rho=i/2000.;
            const auto p=profile.profile(rho);
            reference << rho << ',' << p[0] << ',' << p[2] << ',' << p[1] << ',' << p[3] << '\n';
        }
        // This L2 error is in the global (s,r) coordinate field, not an
        // isometric Euclidean rendering of the hyperbolic target.
        double error2=0.;
        const auto& quadrature=*QF_Simplex<R3>(2);
        std::array<double,4> buffer{};
        RNMK_ basis(buffer.data(),4,1,1);
        for (int k=0;k<mesh->nt;++k) {
            const auto cell=problem.space()[k];
            for (int ip=0;ip<quadrature.n;++ip) {
                const auto& qp=quadrature[ip];
                cell.BF(Fop_D0,qp,basis);
                const auto exact=profile.exact(point<3>(cell.T(qp)));
                Vector value(4,0.);
                for (int j=0;j<4;++j) for (int a=0;a<4;++a)
                    value[a]+=basis(j,0,op_id)*result.coordinates[4*cell(j)+a];
                for (int a=0;a<4;++a) error2+=cell.T.measure()*qp.a*(value[a]-exact[a])*(value[a]-exact[a]);
            }
        }
        std::ofstream summary(prefix+"_summary.csv");
        summary << std::setprecision(17)
            << "status,n,seed,contact_radius,c,tube,tolerance,nodes,cells,iterations,energy,residual,coordinate_l2,max_coordinate_error,min_r,max_r,dual_violation,complementarity,trace_error,active_nodes,contact_cell_volume,interior_reaction_error,max_qp_iterations\n"
            << result.status << ',' << n << ',' << seed << ',' << radius << ',' << slope << ',' << tube << ','
            << options.tolerance << ',' << mesh->nv << ',' << mesh->nt << ',' << result.iterations << ','
            << result.energy << ',' << result.tangent_residual+result.normal_residual << ',' << std::sqrt(error2) << ','
            << max_error << ',' << minr << ',' << maxr << ',' << dual << ',' << complementarity << ','
            << trace_error << ',' << active << ',' << contact_volume << ',' << interior_reaction_error << ','
            << options.max_qp_iterations << '\n';
        if (!nodes || !cells || !reference || !summary) throw std::runtime_error("Failed to write horosphere diagnostics");
        std::cout << result.status << "; coordinate L2=" << std::sqrt(error2) << "; active nodes=" << active << '\n';
        return result.converged ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "obstacle_map_horosphere: " << e.what() << '\n'; return 2;
    }
}
