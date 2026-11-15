#include "../example/obstacle_map/horosphere.hpp"
#include "../example/obstacle_map/horosphere_mesh.hpp"
#include <iostream>
#include <map>

namespace {
void check_ball_mesh() {
    using cutfem::obstacle::examples::horosphere_ball_mesh_3d;
    // Regression at a resolution where the old smooth cube-to-ball mapping
    // produced inverted boundary cells. Check conformity independently of
    // the generator's own adjacency assembly and volume guard.
    const auto mesh=horosphere_ball_mesh_3d(24);
    using Face=std::array<int,3>;
    std::map<Face,int> incidence;
    double volume=0.;
    for (int k=0;k<mesh->nt;++k) {
        const auto& tet=(*mesh)[k];
        Vertex3* vertices[4];
        for (int j=0;j<4;++j) vertices[j]=tet.vertices[j];
        const double signed_volume=DataTet::mesure(vertices);
        if (!(signed_volume>0.)) throw std::runtime_error("Inverted fitted-ball element");
        volume+=signed_volume;
        for (int omitted=0;omitted<4;++omitted) {
            Face face{}; int j=0;
            for (int v=0;v<4;++v) if (v!=omitted) face[j++]=(*mesh)(tet[v]);
            std::sort(face.begin(),face.end());
            ++incidence[face];
        }
    }
    for (int k=0;k<mesh->nbe;++k) {
        const auto& triangle=mesh->be(k);
        Face face{};
        for (int j=0;j<3;++j) {
            face[j]=(*mesh)(triangle[j]);
            const auto& x=(*mesh)(face[j]);
            if (std::abs(std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])-1.)>1e-12)
                throw std::runtime_error("Fitted-ball boundary vertex is not on the unit sphere");
        }
        std::sort(face.begin(),face.end());
        const auto found=incidence.find(face);
        if (found==incidence.end() || found->second!=1)
            throw std::runtime_error("Fitted-ball boundary facets do not match the tetrahedra");
        incidence.erase(found);
    }
    for (const auto& [face,count]:incidence) if (count!=2)
        throw std::runtime_error("Fitted ball has an internal crack or nonmanifold facet");
    if (!(volume>4.17 && volume<4.*std::acos(-1.)/3.))
        throw std::runtime_error("Fitted-ball volume is inconsistent with a unit ball");
}
}

int main() {
    using namespace cutfem::obstacle;
    using namespace cutfem::obstacle::examples;
    using cutfem::obstacle::Vector;
    try {
        check_ball_mesh();
        const HorosphereProfile profile;
        const HorosphereProfile finer(.5,1./std::sqrt(3.),1.,8192);
        for (double rho : {.0,.3,.5,.61,.73,.87,1.}) {
            const auto p=profile.profile(rho), f=finer.profile(rho);
            for (int i=0;i<4;++i) if (std::abs(p[i]-f[i])>2e-10)
                throw std::runtime_error("Radial IVP reference did not stabilize under step refinement");
            if (p.r<0. || p.dr<0.) throw std::runtime_error("Invalid radial distance profile");
        }
        double ode_error=0.;
        for (double rho : {.62,.75,.93}) {
            const double h=1e-5;
            const auto p=profile.profile(rho), plus=profile.profile(rho+h), minus=profile.profile(rho-h);
            const double fpp=(plus.df-minus.df)/(2*h), rpp=(plus.dr-minus.dr)/(2*h);
            const double f_rhs=2*p.f/(rho*rho)-(2/rho+2*p.dr)*p.df;
            const double r_rhs=std::exp(2*p.r)*(p.df*p.df+2*p.f*p.f/(rho*rho))-2*p.dr/rho;
            ode_error=std::max({ode_error,std::abs(fpp-f_rhs),std::abs(rpp-r_rhs)});
        }
        if (ode_error>1e-7) throw std::runtime_error("Radial reference does not satisfy the coupled equations");
        const auto target=profile.chart(.2);
        Vector G; std::vector<Vector> dG;
        const Vector q{.1,-.2,.3,.11};
        target.metric(q,G,dG);
        const double h=1e-6;
        auto p=q,m=q; p[3]+=h; m[3]-=h;
        Vector gp,gm; std::vector<Vector> unused;
        target.metric(p,gp,unused); target.metric(m,gm,unused);
        for (int i=0;i<16;++i) if (std::abs((gp[i]-gm[i])/(2*h)-dG[3][i])>1e-8)
            throw std::runtime_error("Hyperbolic metric derivative is inconsistent");
        const auto mesh=horosphere_ball_mesh_3d(6);
        const ObstacleMapProblem<3> problem(mesh,target);
        ObstacleMapOptions options;
        options.tolerance=1e-6;
        options.max_qp_iterations=3000;
        const auto trace=[profile](const Point<3>& x){return profile.exact(x);};
        const auto result=problem.solve(trace,[profile](const Point<3>& x){return profile.initial(x,2);},options);
        if (!result.converged || result.tangent_residual+result.normal_residual>options.tolerance)
            throw std::runtime_error("Horosphere example failed to converge");
        if (!problem.feasible(result.coordinates)) throw std::runtime_error("Returned map is infeasible");
        int contact=0;
        for (int i=0;i<mesh->nv;++i) {
            if (problem.isBoundary(i)) {
                const auto value=trace(point<3>((*mesh)(i)));
                for (int j=0;j<4;++j) if (value[j]!=result.coordinates[4*i+j])
                    throw std::runtime_error("Boundary trace changed");
            } else {
                contact+=result.contact[i];
                if (result.reaction[i]<-options.tolerance) throw std::runtime_error("Negative obstacle multiplier");
            }
        }
        if (!contact) throw std::runtime_error("The numerical map did not develop contact");
        std::cout << "Radial ODE error=" << ode_error << "; solve residual="
            << result.tangent_residual+result.normal_residual << "; contact nodes=" << contact << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
