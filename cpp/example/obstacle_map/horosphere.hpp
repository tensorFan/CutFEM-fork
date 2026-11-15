#pragma once

#include "benchmarks.hpp"

namespace cutfem::obstacle::examples {

// Hyperbolic four-space of sectional curvature -kappa^2, restricted to the
// exterior of a horoball. In global coordinates q=(s0,s1,s2,r), r>=0,
// G=diag(exp(2*kappa*r),exp(2*kappa*r),exp(2*kappa*r),1).
// Vertical unit-speed geodesics realize distance to the obstacle everywhere.
// The embedding callback gives upper-half-space coordinates (s,exp(-kappa*r)/kappa),
// equipped with the HYPERBOLIC metric (ds^2+dz^2)/(kappa*z)^2. Its Euclidean
// metric is not the target metric; referenceL2 is only a coordinate diagnostic.
struct HorosphereGeometry {
    double kappa = 1.;

    explicit HorosphereGeometry(double curvature_scale = 1.) : kappa(curvature_scale) {
        if (!(kappa > 0.) || !std::isfinite(kappa))
            throw std::invalid_argument("Horosphere curvature scale must be finite and positive");
    }

    TargetChart chart(double tube_radius = .03) const {
        detail::require_tube(tube_radius);
        if (!(tube_radius < 4. / kappa))
            throw std::invalid_argument("Horosphere collar exceeds the numerical chart certificate");
        TargetChart target;
        target.dimension = 4;
        target.tube_radius = tube_radius;
        target.metric = [scale = kappa](const Vector &q, Vector &G,
                                       std::vector<Vector> &derivative) {
            detail::require_coordinates(q, 4);
            const double warp = std::exp(2. * scale * q[3]);
            G.assign(16, 0.);
            derivative.assign(4, Vector(16, 0.));
            for (int j = 0; j < 3; ++j) {
                G[5*j] = warp;
                derivative[3][5*j] = 2. * scale * warp;
            }
            G[15] = 1.;
        };
        target.embed = [scale = kappa](const Vector &q) {
            detail::require_coordinates(q, 4);
            return Vector{q[0], q[1], q[2], std::exp(-scale*q[3]) / scale};
        };
        target.admissible_simplex = [scale = kappa](const std::vector<Vector> &vertices) {
            if (!detail::finite_feasible_vertices(vertices, 4)) return false;
            // The chart itself is global. This convex strip is a numerical
            // guard against extreme exponentials, not an additional obstacle;
            // a reported experiment must remain strictly below the guard.
            for (const auto &q : vertices) if (!(q[3] < 4. / scale)) return false;
            return true;
        };
        return target;
    }
};

// A continuum radial reference, obtained by a regular IVP starting at a
// prescribed contact radius. Its coupled equations outside contact are
//   f'' = 2*f/rho^2 - (2/rho + 2*kappa*R')*f',
//   R'' = kappa*exp(2*kappa*R)*(f'^2 + 2*f^2/rho^2) - 2*R'/rho.
// Inside contact f=c*rho and R=0. The map is q=(f(rho)*x/rho,R(rho));
// its free boundary is the round source sphere rho=a and its contact reaction
// is 3*kappa*c^2. RK4 integration and cubic Hermite evaluation are independent
// of the finite element solver. This class does not certify FEM minimality.
class HorosphereProfile {
  public:
    struct State {
        double f, df, r, dr;

        double operator[](std::size_t coordinate) const {
            switch (coordinate) {
                case 0: return f;
                case 1: return df;
                case 2: return r;
                case 3: return dr;
                default: throw std::out_of_range("Radial profile state has four components");
            }
        }
    };

    explicit HorosphereProfile(double contact_radius = .5, double contact_slope = 1./std::sqrt(3.),
                               double curvature_scale = 1., int integration_steps = 4096)
        : a_(contact_radius), c_(contact_slope), kappa_(curvature_scale),
          steps_(integration_steps) {
        if (!(a_ > 0. && a_ < 1.) || !(c_ > 0.) || !std::isfinite(c_) ||
            !(kappa_ > 0.) || !std::isfinite(kappa_) || steps_ < 32)
            throw std::invalid_argument("Radial horosphere profile requires 0<a<1, c>0, "
                                        "kappa>0, and at least 32 integration steps");
        if (steps_ % 2) ++steps_; // Simpson's rule in continuumEnergy().
        step_ = (1. - a_) / steps_;
        states_.reserve(steps_ + 1);
        states_.push_back({c_*a_, c_, 0., 0.});
        for (int j = 0; j < steps_; ++j) {
            const double rho = a_ + j*step_;
            const auto y = states_.back();
            const auto k1 = derivative(rho, y);
            const auto k2 = derivative(rho + .5*step_, add(y, k1, .5*step_));
            const auto k3 = derivative(rho + .5*step_, add(y, k2, .5*step_));
            const auto k4 = derivative(rho + step_, add(y, k3, step_));
            const State next{
                y.f + step_*(k1.f + 2*k2.f + 2*k3.f + k4.f)/6.,
                y.df + step_*(k1.df + 2*k2.df + 2*k3.df + k4.df)/6.,
                y.r + step_*(k1.r + 2*k2.r + 2*k3.r + k4.r)/6.,
                y.dr + step_*(k1.dr + 2*k2.dr + 2*k3.dr + k4.dr)/6.
            };
            if (!std::isfinite(next.f) || !std::isfinite(next.df) ||
                !std::isfinite(next.r) || !std::isfinite(next.dr) ||
                !(next.df > 0.) || next.r < 0. || !(next.r < 4. / kappa_))
                throw std::invalid_argument("Radial horosphere IVP leaves the numerical chart certificate");
            states_.push_back(next);
        }
    }

    double contactRadius() const { return a_; }
    double contactSlope() const { return c_; }
    double curvatureScale() const { return kappa_; }
    double boundarySlope() const { return states_.back().f; }
    double boundaryDistance() const { return states_.back().r; }
    double contactReaction() const { return 3. * kappa_ * c_ * c_; }
    int integrationSteps() const { return steps_; }
    TargetChart chart(double tube_radius = .03) const {
        return HorosphereGeometry(kappa_).chart(tube_radius);
    }

    State profile(double rho) const {
        if (!(rho >= 0.) || !std::isfinite(rho) || rho > 1. + 1e-12)
            throw std::invalid_argument("Radial reference is defined on the closed unit ball");
        if (rho <= a_) return {c_*rho, c_, 0., 0.};
        if (rho >= 1.) return states_.back();
        const double position = (rho - a_) / step_;
        const int cell = std::min(steps_ - 1, static_cast<int>(position));
        const double t = position - cell;
        const auto &left = states_[cell];
        const auto &right = states_[cell+1];
        const auto f = hermite(left.f, left.df, right.f, right.df, t, step_);
        const auto r = hermite(left.r, left.dr, right.r, right.dr, t, step_);
        return {f.first, f.second, std::max(0., r.first), r.second};
    }

    Vector reference(const Point<3> &x) const {
        const double rho = radius(x);
        const auto value = profile(rho);
        const double scale = rho > 0. ? value.f / rho : c_;
        return {scale*x[0], scale*x[1], scale*x[2], value.r};
    }

    Vector boundary(const Point<3> &x) const { return reference(x); }
    Vector exact(const Point<3> &x) const { return reference(x); }

    Vector initial(const Point<3> &x, int seed = 0) const {
        // Strictly positive normal coordinates, with no supplied contact set.
        // A small nonsymmetric tangential perturbation exercises recovery of
        // the radial solution; the solver overwrites prescribed boundary nodes.
        const double slope = boundarySlope();
        const double phase = .73*seed;
        return {slope*x[0] + .015*std::sin(1.3*x[1] + .4*x[2] + phase),
                slope*x[1] + .015*std::sin(.7*x[0] - .9*x[2] - phase),
                slope*x[2] + .015*std::sin(1.1*x[0] + .6*x[1] + phase),
                boundaryDistance() + .015*(2. + std::cos(x[0] + x[1] + phase))};
    }

    double reaction(const Point<3> &x) const {
        return radius(x) < a_ ? contactReaction() : 0.;
    }

    // Invert the monotone tangential radius f. This locates the reference
    // source radius from a target radius without a coordinate singularity.
    double inverseTangentialRadius(double f) const {
        if (!(f >= 0.) || !std::isfinite(f) || f > boundarySlope() + 1e-12)
            throw std::invalid_argument("Tangential radius leaves the radial reference image");
        if (f <= c_*a_) return f / c_;
        double lower = a_, upper = 1.;
        for (int j = 0; j < 56; ++j) {
            const double middle = .5*(lower + upper);
            if (profile(middle).f < f) lower = middle;
            else upper = middle;
        }
        return .5*(lower + upper);
    }

    // Half Dirichlet energy on the exact unit ball, not the fitted polyhedron.
    double continuumEnergy() const {
        double integral = 0.;
        for (int j = 0; j <= steps_; ++j) {
            const double rho = a_ + j*step_;
            const auto &value = states_[j];
            const double density = std::exp(2*kappa_*value.r) *
                (rho*rho*value.df*value.df + 2*value.f*value.f) +
                rho*rho*value.dr*value.dr;
            const double weight = (j == 0 || j == steps_) ? 1. : (j % 2 ? 4. : 2.);
            integral += weight*density;
        }
        return 2.*std::acos(-1.)*(c_*c_*a_*a_*a_ + integral*step_/3.);
    }

  private:
    double a_, c_, kappa_, step_;
    int steps_;
    std::vector<State> states_;

    static double radius(const Point<3> &x) {
        return std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    }

    State derivative(double rho, const State &value) const {
        return {value.df,
                2*value.f/(rho*rho) - (2/rho + 2*kappa_*value.dr)*value.df,
                value.dr,
                kappa_*std::exp(2*kappa_*value.r)*
                    (value.df*value.df + 2*value.f*value.f/(rho*rho)) - 2*value.dr/rho};
    }

    static State add(const State &y, const State &dy, double h) {
        return {y.f+h*dy.f, y.df+h*dy.df, y.r+h*dy.r, y.dr+h*dy.dr};
    }

    static std::pair<double, double> hermite(double left, double dleft,
                                            double right, double dright,
                                            double t, double h) {
        const double t2 = t*t, t3 = t2*t;
        const double value = (2*t3-3*t2+1)*left + (t3-2*t2+t)*h*dleft +
                             (-2*t3+3*t2)*right + (t3-t2)*h*dright;
        const double derivative = (6*t2-6*t)*left/h + (3*t2-4*t+1)*dleft +
                                  (-6*t2+6*t)*right/h + (3*t2-2*t)*dright;
        return {value, derivative};
    }
};

} // namespace cutfem::obstacle::examples
