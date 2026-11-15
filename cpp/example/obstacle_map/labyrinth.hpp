#pragma once

#include "benchmarks.hpp"

namespace cutfem::obstacle::examples {

// An intrinsic Riemannian target, NOT a Euclidean embedded surface.
// q=(s,t,r), r>=0; G=diag(exp(2*kappa(s,t)*r)*B,1), B SPD.
// Since length >= total variation of r, vertical unit-speed geodesics
// realize distance to r=0 everywhere: this is a global Fermi chart.
struct LabyrinthGeometry {
    double bias = .45;
    double frequency = std::acos(-1.);
    double scale = 2.;
    double twist = .65;
    double height = .025;
    double anisotropy = .25;
    double warp = .12;

    std::array<double,3> curvature(double s, double t) const {
        const double a = frequency*s, b = frequency*t;
        return {bias - std::cos(a)*std::cos(b) + .18*std::sin(a+b)
                     + .12*std::cos(2*a-b+.3),
                frequency*(std::sin(a)*std::cos(b) + .18*std::cos(a+b)
                     - .24*std::sin(2*a-b+.3)),
                frequency*(std::cos(a)*std::sin(b) + .18*std::cos(a+b)
                     + .12*std::sin(2*a-b+.3))};
    }

    TargetChart chart(double tube) const {
        detail::require_tube(tube);
        TargetChart target;
        target.dimension = 3;
        target.tube_radius = tube;
        target.metric = [self=*this](const Vector& q, Vector& G, std::vector<Vector>& dG) {
            detail::require_coordinates(q, 3);
            const auto k = self.curvature(q[0], q[1]);
            const double a = std::exp(2*k[0]*q[2]);
            G = {a, .22*a, 0., .22*a, 1.25*a, 0., 0., 0., 1.};
            dG.assign(3, Vector(9, 0.));
            for (int c=0; c<3; ++c) {
                const double factor = 2*(c==2 ? k[0] : q[2]*k[c+1]);
                for (int j : {0,1,3,4}) dG[c][j] = factor*G[j];
            }
        };
        // Coordinate visualization only; Euclidean distances in this output
        // are not intrinsic target distances and are never used as errors.
        target.embed = [](const Vector& q) { return q; };
        target.admissible_simplex = [](const std::vector<Vector>& vertices) {
            // This convex prism avoids extreme exponentials. It is not an
            // extra variational obstacle; experiments must remain below r=1.
            if (!detail::finite_feasible_vertices(vertices, 3)) return false;
            for (const auto& q : vertices) if (q[2] > 1.) return false;
            return true;
        };
        return target;
    }

    Vector boundary(const Point<3>& x) const {
        const double angle = twist*x[2];
        return {scale*(std::cos(angle)*x[0]-std::sin(angle)*x[1])+.13*x[2],
                scale*(std::sin(angle)*x[0]+std::cos(angle)*x[1])-.09*x[2],
                height*(1.+.12*x[0]+.08*x[1]*x[2])};
    }

    SourceMetric<3> source(const Point<3>& x) const {
        // g^{-1}=I+a v v^T; determinant lemma gives consistent dV_g.
        const Point<3> v{std::cos(1.1*x[2]), std::sin(1.1*x[2]), .35};
        SourceMetric<3> g;
        double norm2=0.;
        for (int i=0; i<3; ++i) {
            norm2 += v[i]*v[i];
            for (int j=0; j<3; ++j) g.inverse[i][j] += anisotropy*v[i]*v[j];
        }
        g.density = 1./std::sqrt(1.+anisotropy*norm2);
        return g;
    }

    std::shared_ptr<Mesh3> mesh(int n, bool ball) const {
        auto mesh = ball ? unit_ball_mesh_3d(n) : box_mesh_3d(n);
        // A smooth shear, invertible globally: z unchanged, y unchanged,
        // x translated by warp*y*z. It changes the fitted source domain.
        for (int i=0; i<mesh->nv; ++i) mesh->v(i)[0] += warp*mesh->v(i)[1]*mesh->v(i)[2];
        mesh->mes=mesh->mesb=0.;
        for (int k=0; k<mesh->nt; ++k) {
            auto& cell=mesh->t(k);
            cell.mes=DataTet::mesure(cell.vertices.data());
            if (!(cell.mes>0.)) throw std::runtime_error("Inverted experimental source cell");
            mesh->mes+=cell.mes;
        }
        for (int k=0; k<mesh->nbe; ++k) {
            auto& face=mesh->be(k);
            face.mes=DataTriangle3::mesure(face.vertices.data());
            mesh->mesb+=face.mes;
        }
        return mesh;
    }
};

} // namespace cutfem::obstacle::examples
