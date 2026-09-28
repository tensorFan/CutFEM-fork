#include "../problem/obstacleTarget.hpp"
#include <iostream>

using namespace cutfem::obstacle;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class Function> void rejected(Function operation) {
    bool invalid = false;
    try { operation(); } catch (const std::invalid_argument&) { invalid = true; }
    require(invalid, "Malformed geometry was accepted");
}

// Noncircular target: inward Fermi collar of y = beta*x^2, beta = .3.
EmbeddingJet parabolaJet(const Vector& q, double k) {
    const double s = q[0], r = q[1], h = k*s;
    const double w = std::sqrt(1.+h*h), w3 = w*w*w, w5 = w3*w*w;
    const double nx = -h/w, ny = 1./w;
    const double dx = -k/w3, dy = -k*h/w3;
    const double ddx = 3.*k*k*h/w5, ddy = -k*k/w3 + 3.*k*k*h*h/w5;
    return {{s+r*nx, .5*k*s*s+r*ny},
            {1.+r*dx, nx, k*s+r*dy, ny},
            {r*ddx, dx, dx, 0., k+r*ddy, dy, dy, 0.}};
}
EmbeddingJet parabola(const Vector& q) { return parabolaJet(q, .6); }

bool collar(const std::vector<Vector>& vertices) {
    // This coordinate box is convex and lies below the focal distance 1/.6.
    // On this box y < .55, so the normal-foot equation has derivative
    // 1+.54*s^2-.6*y > 0; consequently the normal foot is unique.
    for (const auto& q : vertices)
        if (std::abs(q[0]) > 1. || q[1] < 0. || q[1] > .25) return false;
    return true;
}

void derivativeCheck(const TargetChart& target, const Vector& q) {
    Vector G;
    std::vector<Vector> dG;
    target.metric(q, G, dG);
    const int m = target.dimension;
    const double h = 2e-6;
    std::vector<Vector> numerical_jacobian(m);
    for (int a = 0; a < m; ++a) {
        Vector plus = q, minus = q, Gplus, Gminus;
        plus[a] += h; minus[a] -= h;
        const auto Fplus = target.embed(plus), Fminus = target.embed(minus);
        numerical_jacobian[a].resize(Fplus.size());
        for (std::size_t p = 0; p < Fplus.size(); ++p)
            numerical_jacobian[a][p] = (Fplus[p]-Fminus[p])/(2.*h);
        std::vector<Vector> unused;
        target.metric(plus, Gplus, unused);
        target.metric(minus, Gminus, unused);
        for (int ab = 0; ab < m*m; ++ab)
            require(std::abs((Gplus[ab]-Gminus[ab])/(2.*h)-dG[a][ab]) < 2e-9,
                    "Metric derivative does not differentiate the pullback metric");
    }
    for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) {
        double numerical = 0.;
        for (std::size_t p = 0; p < numerical_jacobian[a].size(); ++p)
            numerical += numerical_jacobian[a][p]*numerical_jacobian[b][p];
        require(std::abs(numerical-G[a*m+b]) < 2e-9, "Metric disagrees with embedding");
    }
}
} // namespace

int main() {
    try {
        const auto target = embeddedTargetChart(2, 2, .25, parabola, collar);
        for (double s : {-.8, -.2, .0, .4, .9}) for (double r : {.02, .11, .23}) {
            Vector G; std::vector<Vector> dG;
            target.metric({s,r}, G, dG);
            const double w = std::sqrt(1.+.36*s*s), factor = 1.-r*.6/(w*w*w);
            require(std::abs(G[0]-w*w*factor*factor) < 1e-13 &&
                    std::abs(G[1]) < 1e-13 && std::abs(G[3]-1.) < 1e-13,
                    "Parabola does not satisfy the Fermi metric identities");
            derivativeCheck(target, {s,r});
        }
        require(target.admissible_simplex({{-.8,.02},{.9,.1},{.1,.24}}), "Valid collar rejected");
        require(!target.admissible_simplex({{0.,.3}}), "Certificate was bypassed");
        require(!target.admissible_simplex({{0.,-.1}}), "Negative normal coordinate accepted");

        // A curved boundary surface in R^3, extruded along its constant normal
        // into R^4. Its two tangential coordinates have a non-diagonal metric.
        const auto surface = embeddedTargetChart(3, 4, .25, [](const Vector& q) {
            const double s = q[0], t = q[1];
            EmbeddingJet jet{{s,t,s*t,q[2]}, {1.,0.,0., 0.,1.,0., t,s,0., 0.,0.,1.}, Vector(36,0.)};
            jet.hessian[(2*3+0)*3+1] = jet.hessian[(2*3+1)*3+0] = 1.;
            return jet;
        }, [](const std::vector<Vector>&) { return true; });
        derivativeCheck(surface, {.3,.4,.15});
        Vector G; std::vector<Vector> dG;
        surface.metric({.3,.4,.15}, G, dG);
        require(std::abs(G[1]-.12) < 1e-14, "Tangential metric coupling was discarded");

        // Concave parabola: obstacle contact is favored by the boundary data.
        // This five-node solve also exercises the adapter through FEM assembly.
        SimplexMesh<2> mesh{{{-1.,-1.},{1.,-1.},{1.,1.},{-1.,1.},{0.,0.}},
                            {{0,1,4},{1,2,4},{2,3,4},{3,0,4}},
                            {true,true,true,true,false}};
        auto concave = embeddedTargetChart(2,2,.25,
            [](const Vector& q) { return parabolaJet(q,-.6); },
            [](const std::vector<Vector>& vertices) {
                // For beta<0 there is no inward focal point. In this box
                // y>=-.3 and the normal-foot derivative is at least .82.
                return collar(vertices);
            });
        const ObstacleMapProblem<2> problem(std::move(mesh),std::move(concave));
        const auto result = problem.solve(
            [](const Point<2>& x) { return Vector{.2*x[0],0.}; },
            [](const Point<2>& x) { return Vector{.2*x[0]+.03,.05}; });
        require(result.converged && problem.feasible(result.coordinates) &&
                result.tangent_residual+result.normal_residual <= 1e-7,
                "Parabola FEM solve did not reach feasible stationarity");
        require(result.contact[4] && result.reaction[4] > .01,
                "Concave parabola did not produce positive obstacle contact reaction");

        rejected([&] { embeddedTargetChart(2,1,.25,parabola,collar); });
        rejected([&] { embeddedTargetChart(2,2,0.,parabola,collar); });
        rejected([&] { embeddedTargetChart(2,2,.25,{},collar); });
        rejected([&] { embeddedTargetChart(2,2,.25,parabola,{}); });
        rejected([&] { target.embed({0.}); });
        rejected([&] { target.embed({0.,std::numeric_limits<double>::infinity()}); });
        for (int defect = 0; defect < 7; ++defect) {
            auto malformed = embeddedTargetChart(2,2,.25,[defect](const Vector& q) {
                auto jet = parabola(q);
                if (defect == 0) jet.value.pop_back();
                if (defect == 1) jet.jacobian.pop_back();
                if (defect == 2) jet.hessian.pop_back();
                if (defect == 3) jet.hessian[1] += .1;
                if (defect == 4) jet.hessian[0] = std::numeric_limits<double>::quiet_NaN();
                if (defect == 5) jet.jacobian[0] = std::numeric_limits<double>::infinity();
                if (defect == 6) std::fill(jet.jacobian.begin(), jet.jacobian.end(), 0.);
                return jet;
            },collar);
            rejected([&] { malformed.metric({.2,.1},G,dG); });
        }
        std::cout << "Embedding target tests passed (parabola collar and coupled surface metric).\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
