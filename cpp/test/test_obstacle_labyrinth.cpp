#include "../example/obstacle_map/labyrinth.hpp"
#include <iostream>

using namespace cutfem::obstacle;
using namespace cutfem::obstacle::examples;

int main() {
    using cutfem::obstacle::Vector;
    try {
        const LabyrinthGeometry geometry;
        const auto target=geometry.chart(.05);
        double max_metric_error=0.;
        for (const Vector q : {Vector{.13,-.27,.07}, Vector{-.6,.51,.24}, Vector{1.2,-.91,.42}}) {
            Vector G;
            std::vector<Vector> derivatives;
            target.metric(q,G,derivatives);
            for (int c=0; c<3; ++c) {
                auto p=q, m=q;
                const double h=1e-6;
                p[c]+=h; m[c]-=h;
                Vector gp,gm; std::vector<Vector> unused;
                target.metric(p,gp,unused); target.metric(m,gm,unused);
                for (int j=0; j<9; ++j)
                    max_metric_error=std::max(max_metric_error,std::abs((gp[j]-gm[j])/(2*h)-derivatives[c][j]));
            }
        }
        if (max_metric_error>2e-8) throw std::runtime_error("Incorrect target metric derivatives");
        const auto metric=geometry.source({.2,.3,-.4});
        const auto& a=metric.inverse;
        const double det=a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
            -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
            +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
        if (std::abs(det*metric.density*metric.density-1.)>1e-13)
            throw std::runtime_error("Inconsistent source density");
        auto mesh=geometry.mesh(4,false);
        if (std::abs(mesh->mes-8.)>1e-12) throw std::runtime_error("Shear changed source volume");
        ObstacleMapProblem<3> problem(mesh,target,[geometry](const Point<3>& x){return geometry.source(x);});
        auto q=problem.interpolate([](const Point<3>& x){return Vector{.4*x[0],.5*x[1],.12+.02*x[2]};});
        Vector direction(q.size());
        for (std::size_t i=0; i<q.size(); ++i) direction[i]=std::sin(.71*i);
        const auto assembly=problem.assemble(q);
        double exact=0.;
        for (std::size_t i=0; i<q.size(); ++i) exact+=direction[i]*assembly.gradient[i];
        auto p=q,m=q;
        const double h=1e-6;
        for (std::size_t i=0; i<q.size(); ++i) { p[i]+=h*direction[i]; m[i]-=h*direction[i]; }
        const double numerical=(problem.assemble(p,false).energy-problem.assemble(m,false).energy)/(2*h);
        if (std::abs(exact-numerical)>1e-8*(1+std::abs(exact)))
            throw std::runtime_error("Experiment energy gradient failed directional check");
        std::cout << "Metric derivative error=" << max_metric_error
                  << "; assembled directional error=" << std::abs(exact-numerical) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
