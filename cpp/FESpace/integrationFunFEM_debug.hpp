/*
This file is part of CutFEM-Library.

CutFEM-Library is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

CutFEM-Library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
CutFEM-Library. If not, see <https://www.gnu.org/licenses/>.
*/
#ifndef INTEGRATION_FUNFEM_HPP_
#define INTEGRATION_FUNFEM_HPP_

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <list>
#include <memory>

#include "expression.hpp"
#include "macroElement.hpp"

namespace IntegrationFunFEMDetail {

#ifndef INTEGRATION_FUNFEM_DEBUG
#define INTEGRATION_FUNFEM_DEBUG 1
#endif

#ifndef INTEGRATION_FUNFEM_DEBUG_LIMIT
#define INTEGRATION_FUNFEM_DEBUG_LIMIT 2000
#endif

#ifndef INTEGRATION_FUNFEM_DEBUG_EVERY
#define INTEGRATION_FUNFEM_DEBUG_EVERY 1
#endif

inline bool debug_enabled() {
#if INTEGRATION_FUNFEM_DEBUG
    return true;
#else
    return false;
#endif
}

inline void debug_line(const char *msg) {
#if INTEGRATION_FUNFEM_DEBUG
    std::cerr << "[integrationFunFEM] " << msg << std::endl;
#endif
}

inline bool debug_should_print(long long counter) {
#if INTEGRATION_FUNFEM_DEBUG
    if (counter < INTEGRATION_FUNFEM_DEBUG_LIMIT)
        return (counter % INTEGRATION_FUNFEM_DEBUG_EVERY) == 0;
#endif
    return false;
}

inline double mpi_sum(double value) {
#ifdef USE_MPI
    double received = 0.;
    MPIcf::AllReduce(value, received, MPI_SUM);
    return received;
#else
    return value;
#endif
}

template <typename M>
std::shared_ptr<const ExpressionVirtual>
make_id_expression(const FunFEM<M> &fh, int component) {
    return std::make_shared<const ExpressionFunFEM<M>>(fh, component, op_id);
}

template <typename M, typename Predicate, typename Visitor>
double for_each_active_cut_quadrature(const ActiveMesh<M> &Th, int itq,
                                      Predicate accept_element,
                                      Visitor visit) {
    typedef M Mesh;
    typedef typename Mesh::Element Element;
    typedef GFESpace<Mesh> FESpace;
    typedef typename FESpace::FElement FElement;
    typedef typename FElement::QF QF;
    typedef typename FElement::Rd Rd;
    typedef typename QF::QuadraturePoint QuadraturePoint;

    const QF &qf(*QF_Simplex<typename FElement::RdHat>(5));
    double value = 0.;
    long long debug_counter = 0;

    if (debug_enabled()) {
        std::cerr << "[integrationFunFEM] ENTER for_each_active_cut_quadrature"
                  << " itq=" << itq
                  << " first=" << Th.first_element()
                  << " last=" << Th.last_element()
                  << " step=" << Th.next_element()
                  << " nb_dom=" << Th.get_nb_domain()
                  << std::endl;
    }

    for (int k = Th.first_element(); k < Th.last_element(); k += Th.next_element()) {
        if (debug_should_print(debug_counter)) {
            std::cerr << "[integrationFunFEM] element-begin"
                      << " counter=" << debug_counter
                      << " k=" << k
                      << " itq=" << itq
                      << std::endl;
        }

        if (Th.isInactive(k, itq)) {
            if (debug_should_print(debug_counter)) {
                std::cerr << "[integrationFunFEM] element-skip inactive"
                          << " k=" << k << " itq=" << itq << std::endl;
            }
            ++debug_counter;
            continue;
        }

        if (!accept_element(k)) {
            if (debug_should_print(debug_counter)) {
                std::cerr << "[integrationFunFEM] element-skip rejected"
                          << " k=" << k << std::endl;
            }
            ++debug_counter;
            continue;
        }

        const int domain = Th.get_domain_element(k);
        const int kb = Th.idxElementInBackMesh(k);

        if (debug_should_print(debug_counter)) {
            std::cerr << "[integrationFunFEM] element-accepted"
                      << " counter=" << debug_counter
                      << " active_k=" << k
                      << " back_kb=" << kb
                      << " domain=" << domain
                      << std::endl;
        }

        const Cut_Part<Element> cutK(Th.get_cut_part(k, itq));

        int subcell = 0;
        for (auto it = cutK.element_begin(); it != cutK.element_end(); ++it, ++subcell) {
            const R measure = cutK.measure(it);

            if (debug_should_print(debug_counter)) {
                std::cerr << "[integrationFunFEM] subcell-begin"
                          << " active_k=" << k
                          << " back_kb=" << kb
                          << " domain=" << domain
                          << " subcell=" << subcell
                          << " measure=" << measure
                          << std::endl;
            }

            for (int ipq = 0; ipq < qf.getNbrOfQuads(); ++ipq) {
                QuadraturePoint ip(qf[ipq]);
                Rd mip = cutK.mapToPhysicalElement(it, ip);
                const R weight = measure * ip.getWeight();

                if (debug_should_print(debug_counter)) {
                    std::cerr << "[integrationFunFEM] before visit/eval"
                              << " counter=" << debug_counter
                              << " active_k=" << k
                              << " back_kb=" << kb
                              << " domain=" << domain
                              << " subcell=" << subcell
                              << " ipq=" << ipq
                              << " weight=" << weight
                              << " mip=" << mip
                              << std::endl;
                }

                const double local_value = visit(k, kb, domain, mip);

                if (debug_should_print(debug_counter)) {
                    std::cerr << "[integrationFunFEM] after visit/eval"
                              << " counter=" << debug_counter
                              << " active_k=" << k
                              << " back_kb=" << kb
                              << " domain=" << domain
                              << " subcell=" << subcell
                              << " ipq=" << ipq
                              << " local_value=" << local_value
                              << " contribution=" << weight * local_value
                              << std::endl;
                }

                value += weight * local_value;
                ++debug_counter;
            }
        }
    }

    if (debug_enabled()) {
        std::cerr << "[integrationFunFEM] EXIT for_each_active_cut_quadrature"
                  << " itq=" << itq
                  << " accumulated=" << value
                  << " evaluations=" << debug_counter
                  << std::endl;
    }

    return value;
}

template <typename M>
double active_expression_integral_local(const ActiveMesh<M> &Th,
                                        const std::shared_ptr<const ExpressionVirtual> &fh,
                                        int itq) {
    if (debug_enabled()) {
        std::cerr << "[integrationFunFEM] ENTER active_expression_integral_local(all domains)"
                  << " itq=" << itq
                  << " expr_ptr=" << fh.get()
                  << std::endl;
    }
    return for_each_active_cut_quadrature(
        Th, itq,
        [](int) { return true; },
        [&](int k, int, int, const typename GFESpace<M>::FElement::Rd &mip) {
            // Active-volume quadrature is already parametrized by the active
            // mesh element index k.  Do not round-trip through the background
            // mesh with evalOnBackMesh(kb, domain, ...): that can map to -1 or
            // to the wrong active space for cut functions.  This follows the
            // convention used in normsFunFEM.hpp/cpp.
            return fh->eval(k, mip);
        });
}

template <typename M>
double active_expression_integral_local(const ActiveMesh<M> &Th,
                                        const std::shared_ptr<const ExpressionVirtual> &fh,
                                        int domain, int itq) {
    if (debug_enabled()) {
        std::cerr << "[integrationFunFEM] ENTER active_expression_integral_local(single domain)"
                  << " requested_domain=" << domain
                  << " itq=" << itq
                  << " expr_ptr=" << fh.get()
                  << std::endl;
    }
    return for_each_active_cut_quadrature(
        Th, itq,
        [&](int k) { return Th.get_domain_element(k) == domain; },
        [&](int k, int, int, const typename GFESpace<M>::FElement::Rd &mip) {
            // See the overload above: active-volume integrals should evaluate
            // directly on the active element index.
            return fh->eval(k, mip);
        });
}

template <typename M>
double active_constant_integral_local(const ActiveMesh<M> &Th, double constant, int itq) {
    return constant * for_each_active_cut_quadrature(
        Th, itq,
        [](int) { return true; },
        [](int, int, int, const typename GFESpace<M>::FElement::Rd &) { return 1.; });
}

template <typename M>
double active_constant_integral_local(const ActiveMesh<M> &Th, double constant,
                                      int domain, int itq) {
    return constant * for_each_active_cut_quadrature(
        Th, itq,
        [&](int k) { return Th.get_domain_element(k) == domain; },
        [](int, int, int, const typename GFESpace<M>::FElement::Rd &) { return 1.; });
}

template <typename M>
double active_time_expression_integral_local(const ActiveMesh<M> &Th, const TimeSlab &In,
                                             const std::shared_ptr<const ExpressionVirtual> &fh,
                                             const QuadratureFormular1d &qTime,
                                             int domain) {
    double value = 0.;

    for (int itq = 0; itq < qTime.n; ++itq) {
        GQuadraturePoint<R1> tq(qTime[itq]);
        const double t = In.mapToPhysicalElement(tq);
        const double time_weight = In.T.measure() * tq.a;

        value += time_weight * for_each_active_cut_quadrature(
            Th, itq,
            [&](int k) { return Th.get_domain_element(k) == domain; },
            [&](int k, int, int, const typename GFESpace<M>::FElement::Rd &mip) {
                // Same rule as the static active-volume integral: use the
                // active mesh element index directly.
                return fh->eval(k, mip, t);
            });
    }

    return value;
}

template <typename Mesh>
double plain_mesh_expression_integral_local(const Mesh &Th,
                                            const std::shared_ptr<const ExpressionVirtual> &fh) {
    typedef typename Mesh::Element Element;
    typedef GFESpace<Mesh> FESpace;
    typedef typename FESpace::FElement FElement;
    typedef typename FElement::QF QF;
    typedef typename FElement::Rd Rd;
    typedef typename QF::QuadraturePoint QuadraturePoint;

    const QF &qf(*QF_Simplex<typename FElement::RdHat>(5));
    double value = 0.;

    for (int k = Th.first_element(); k < Th.last_element(); k += Th.next_element()) {
        const Element &K(Th[k]);
        const R measure = K.measure();

        for (int ipq = 0; ipq < qf.getNbrOfQuads(); ++ipq) {
            QuadraturePoint ip(qf[ipq]);
            Rd mip = K.mapToPhysicalElement(ip);
            const R weight = measure * ip.getWeight();
            value += weight * fh->evalOnBackMesh(k, 0, mip);
        }
    }

    return value;
}

template <typename M>
double interface_funfem_integral_local(FunFEM<M> &fh, const Interface<M> &interface,
                                       int component, double t) {
    typedef GFESpace<M> FESpace;
    typedef typename FESpace::FElement FElement;
    typedef typename FElement::QFB QFB;
    typedef typename FElement::Rd Rd;
    typedef typename QFB::QuadraturePoint QuadraturePoint;

    if (t > -globalVariable::Epsilon && fh.In) {
        assert(fh.In->Pt(0) <= t && t <= fh.In->Pt(1));
    }
    if (t < -globalVariable::Epsilon && fh.In) {
        t = fh.In->Pt(0);
        std::cout << " Use default value In(0) \t -> " << t << std::endl;
    }

    const QFB &qfb(*QF_Simplex<typename FElement::RdHatBord>(5));
    double value = 0.;

    for (int iface = interface.first_element(); iface < interface.last_element();
         iface += interface.next_element()) {
        const int kb = interface.idxElementOfFace(iface);
        const R measure = interface.measure(iface);

        for (int ipq = 0; ipq < qfb.getNbrOfQuads(); ++ipq) {
            QuadraturePoint ip(qfb[ipq]);
            Rd mip = interface.mapToPhysicalFace(iface, (typename FElement::RdHatBord)ip);
            const R weight = measure * ip.getWeight();
            value += weight * fh.evalOnBackMesh(kb, 0, mip, t, component, 0, 0);
        }
    }

    return value;
}

template <typename M>
double time_interface_funfem_integral_local(FunFEM<M> &fh, const TimeSlab &In,
                                            const TimeInterface<M> &gamma, int component) {
    typedef GFESpace<M> FESpace;
    typedef typename FESpace::FElement FElement;
    typedef typename FElement::QFB QFB;
    typedef typename FElement::Rd Rd;
    typedef typename QFB::QuadraturePoint QuadraturePoint;

    const QFB &qfb(*QF_Simplex<typename FElement::RdHatBord>(5));
    double value = 0.;

    for (int it = 0; it < gamma.size(); ++it) {
        const Interface<M> &interface(*gamma(it));
        const QuadratureFormular1d *qTime(gamma.get_quadrature_time());
        GQuadraturePoint<R1> tq((*qTime)[it]);
        const double t = In.mapToPhysicalElement(tq);
        const double time_weight = In.T.measure() * tq.a;

        for (int iface = interface.first_element(); iface < interface.last_element();
             iface += interface.next_element()) {
            const int kb = interface.idxElementOfFace(iface);
            const R measure = interface.measure(iface);

            for (int ipq = 0; ipq < qfb.getNbrOfQuads(); ++ipq) {
                QuadraturePoint ip(qfb[ipq]);
                Rd mip = interface.mapToPhysicalFace(iface, (typename FElement::RdHatBord)ip);
                const R weight = measure * ip.getWeight() * time_weight;
                value += weight * fh.evalOnBackMesh(kb, 0, mip, t, component, 0, 0);
            }
        }
    }

    return value;
}

template <typename M, typename E>
double time_interface_expression_integral_local(const std::shared_ptr<E> &fh,
                                                const TimeSlab &In,
                                                const TimeInterface<M> &gamma) {
    typedef GFESpace<M> FESpace;
    typedef typename FESpace::FElement FElement;
    typedef typename FElement::QFB QFB;
    typedef typename FElement::Rd Rd;
    typedef typename QFB::QuadraturePoint QuadraturePoint;

    const QFB &qfb(*QF_Simplex<typename FElement::RdHatBord>(5));
    double value = 0.;

    for (int it = 0; it < gamma.size(); ++it) {
        const Interface<M> &interface(*gamma(it));
        const QuadratureFormular1d *qTime(gamma.get_quadrature_time());
        GQuadraturePoint<R1> tq((*qTime)[it]);
        const double t = In.mapToPhysicalElement(tq);
        const double time_weight = In.T.measure() * tq.a;

        for (int iface = interface.first_element(); iface < interface.last_element();
             iface += interface.next_element()) {
            const int kb = interface.idxElementOfFace(iface);
            const R measure = interface.measure(iface);
            const Rd normal(-interface.normal(iface));

            for (int ipq = 0; ipq < qfb.getNbrOfQuads(); ++ipq) {
                QuadraturePoint ip(qfb[ipq]);
                Rd mip = interface.mapToPhysicalFace(iface, (typename FElement::RdHatBord)ip);
                const R weight = measure * ip.getWeight() * time_weight;
                value += weight * fh->evalOnBackMesh(kb, 0, mip, t, normal);
            }
        }
    }

    return value;
}

template <typename Mesh>
double boundary_uncut_face_integral_local(const std::shared_ptr<const ExpressionVirtual> &fh,
                                          const ActiveMesh<Mesh> &Th, const TimeSlab &In,
                                          int k, const typename Mesh::BorderElement &BE,
                                          int ifac, const QuadratureFormular1d &qTime,
                                          int itq) {
    typedef typename Mesh::Element Element;
    typedef GFESpace<Mesh> FESpace;
    typedef typename FESpace::FElement FElement;
    typedef typename FElement::QFB QFB;
    typedef typename FElement::Rd Rd;
    typedef typename FElement::RdHatBord RdHatBord;
    typedef typename QFB::QuadraturePoint QuadraturePoint;

    const QFB &qfb(*QF_Simplex<typename FElement::RdHatBord>(5));
    const auto tq = qTime[itq];
    const double t = In.map(tq);
    const double time_weight = tq.a * In.get_measure();
    const int domain = Th.get_domain_element(k);
    const int kb = Th.idxElementInBackMesh(k);
    const Element &K(Th[k]);
    const Rd normal = K.N(ifac);
    const R measure = K.mesureBord(ifac);

    double value = 0.;
    for (int ipq = 0; ipq < qfb.getNbrOfQuads(); ++ipq) {
        QuadraturePoint ip(qfb[ipq]);
        const Rd mip = BE.mapToPhysicalElement((RdHatBord)ip);
        const R weight = measure * ip.getWeight() * time_weight;
        value += weight * fh->evalOnBackMesh(kb, domain, mip, t, normal);
    }

    return value;
}

template <typename Mesh>
double boundary_cut_face_integral_local(const std::shared_ptr<const ExpressionVirtual> &fh,
                                        const ActiveMesh<Mesh> &Th, const TimeSlab &In,
                                        int k, int ifac,
                                        const QuadratureFormular1d &qTime,
                                        int itq) {
    typedef typename Mesh::Element Element;
    typedef GFESpace<Mesh> FESpace;
    typedef typename FESpace::FElement FElement;
    typedef typename FElement::QFB QFB;
    typedef typename FElement::Rd Rd;
    typedef typename FElement::RdHatBord RdHatBord;
    typedef typename QFB::QuadraturePoint QuadraturePoint;

    const QFB &qfb(*QF_Simplex<typename FElement::RdHatBord>(5));
    const auto tq = qTime[itq];
    const double t = In.map(tq);
    const double time_weight = tq.a * In.get_measure();
    const int domain = Th.get_domain_element(k);
    const int kb = Th.idxElementInBackMesh(k);
    const Element &K(Th[k]);
    const Rd normal = K.N(ifac);

    typename Element::Face face;
    const Cut_Part<typename Element::Face> cutFace(Th.get_cut_face(face, k, ifac, itq));

    double value = 0.;
    for (auto it = cutFace.element_begin(); it != cutFace.element_end(); ++it) {
        const R measure = cutFace.measure(it);

        for (int ipq = 0; ipq < qfb.getNbrOfQuads(); ++ipq) {
            QuadraturePoint ip(qfb[ipq]);
            const Rd mip = cutFace.mapToPhysicalElement(it, (RdHatBord)ip);
            const R weight = measure * ip.getWeight() * time_weight;
            value += weight * fh->evalOnBackMesh(kb, domain, mip, t, normal);
        }
    }

    return value;
}

} // namespace IntegrationFunFEMDetail

// -----------------------------------------------------------------------------
// Active mesh volume integration
// -----------------------------------------------------------------------------

template <typename M>
double integral(const ActiveMesh<M> &Th, const std::shared_ptr<const ExpressionVirtual> &fh,
                int domain, int itq) {
    if (IntegrationFunFEMDetail::debug_enabled()) {
        std::cerr << "[integrationFunFEM] ENTER integral(ActiveMesh, shared_ptr<ExpressionVirtual>, domain, itq)"
                  << " domain=" << domain
                  << " itq=" << itq
                  << std::endl;
    }
    double local = IntegrationFunFEMDetail::active_expression_integral_local(Th, fh, domain, itq);
    IntegrationFunFEMDetail::debug_line("REDUCE integral(ActiveMesh, shared_ptr<ExpressionVirtual>, domain, itq)");
    return IntegrationFunFEMDetail::mpi_sum(local);
}

template <typename M>
double integral(const ActiveMesh<M> &Th, const std::shared_ptr<const ExpressionVirtual> &fh,
                int itq) {
    // Integrates over the actual active elements. This avoids assuming that
    // domains are numbered 0,...,Th.get_nb_domain()-1.
    IntegrationFunFEMDetail::debug_line("ENTER integral(ActiveMesh, shared_ptr<ExpressionVirtual>, itq)");
    double local = IntegrationFunFEMDetail::active_expression_integral_local(Th, fh, itq);
    IntegrationFunFEMDetail::debug_line("REDUCE integral(ActiveMesh, shared_ptr<ExpressionVirtual>, itq)");
    return IntegrationFunFEMDetail::mpi_sum(local);
}

template <typename M>
double integral(const ActiveMesh<M> &Th, const std::shared_ptr<const ExpressionVirtual> &fh) {
    return integral(Th, fh, 0);
}

template <typename M>
double integral(const ActiveMesh<M> &Th, const FunFEM<M> &fh, int component,
                int domain, int itq) {
    if (IntegrationFunFEMDetail::debug_enabled()) {
        std::cerr << "[integrationFunFEM] ENTER integral(ActiveMesh, FunFEM, component, domain, itq)"
                  << " component=" << component
                  << " domain=" << domain
                  << " itq=" << itq
                  << std::endl;
    }
    return integral(Th, IntegrationFunFEMDetail::make_id_expression(fh, component), domain, itq);
}

template <typename M>
double integral(const ActiveMesh<M> &Th, const FunFEM<M> &fh, int component,
                int itq) {
    if (IntegrationFunFEMDetail::debug_enabled()) {
        std::cerr << "[integrationFunFEM] ENTER integral(ActiveMesh, FunFEM, component, itq)"
                  << " component=" << component
                  << " itq=" << itq
                  << std::endl;
    }
    return integral(Th, IntegrationFunFEMDetail::make_id_expression(fh, component), itq);
}

template <typename M>
double integral(const ActiveMesh<M> &Th, const FunFEM<M> &fh, int component) {
    if (IntegrationFunFEMDetail::debug_enabled()) {
        std::cerr << "[integrationFunFEM] ENTER integral(ActiveMesh, FunFEM, component)"
                  << " component=" << component
                  << " -> dispatch itq=0"
                  << std::endl;
    }
    return integral(Th, fh, component, 0);
}

template <typename M>
double integral(const ActiveMesh<M> &Th, const double constant, int domain, int itq) {
    return IntegrationFunFEMDetail::mpi_sum(
        IntegrationFunFEMDetail::active_constant_integral_local(Th, constant, domain, itq));
}

template <typename M>
double integral(const ActiveMesh<M> &Th, const double constant) {
    return IntegrationFunFEMDetail::mpi_sum(
        IntegrationFunFEMDetail::active_constant_integral_local(Th, constant, 0));
}

// Direct integration of an analytic function f(Rd point, int component, int domain).
// This is useful for manufactured data and avoids interpolating into a FunFEM first.
template <typename M, typename Function>
double integralFunction(const ActiveMesh<M> &Th, Function f, int component = 0, int itq = 0) {
    typedef typename GFESpace<M>::FElement FElement;
    return IntegrationFunFEMDetail::mpi_sum(
        IntegrationFunFEMDetail::for_each_active_cut_quadrature(
            Th, itq,
            [](int) { return true; },
            [&](int, int, int domain, const typename FElement::Rd &mip) {
                return f(mip, component, domain);
            }));
}

// -----------------------------------------------------------------------------
// Active mesh space-time volume integration
// -----------------------------------------------------------------------------

template <typename M>
double integral(const ActiveMesh<M> &Th, const TimeSlab &In,
                const std::shared_ptr<const ExpressionVirtual> &fh,
                const QuadratureFormular1d &qTime, int domain) {
    return IntegrationFunFEMDetail::mpi_sum(
        IntegrationFunFEMDetail::active_time_expression_integral_local(Th, In, fh, qTime, domain));
}

template <typename M>
double integral(const ActiveMesh<M> &Th, const TimeSlab &In, const FunFEM<M> &fh,
                int component, const QuadratureFormular1d &qTime) {
    std::shared_ptr<const ExpressionVirtual> ui =
        IntegrationFunFEMDetail::make_id_expression(fh, component);

    double value = 0.;
    for (int domain = 0; domain < Th.get_nb_domain(); ++domain) {
        value += IntegrationFunFEMDetail::active_time_expression_integral_local(
            Th, In, ui, qTime, domain);
    }
    return IntegrationFunFEMDetail::mpi_sum(value);
}

// -----------------------------------------------------------------------------
// Plain mesh volume integration
// -----------------------------------------------------------------------------

template <typename Mesh>
double integral(const Mesh &Th, const std::shared_ptr<const ExpressionVirtual> &fh, int /*itq*/) {
    return IntegrationFunFEMDetail::mpi_sum(
        IntegrationFunFEMDetail::plain_mesh_expression_integral_local(Th, fh));
}

template <typename Mesh>
double integral(const Mesh &Th, const std::shared_ptr<const ExpressionVirtual> &fh) {
    return integral(Th, fh, 0);
}

template <typename Mesh>
double integral(const Mesh &Th, const FunFEM<Mesh> &fh, int component) {
    return integral(Th, IntegrationFunFEMDetail::make_id_expression(fh, component));
}

// -----------------------------------------------------------------------------
// Interface integration
// -----------------------------------------------------------------------------

template <typename M>
double integral(FunFEM<M> &fh, const Interface<M> &interface, int component, double t) {
    return IntegrationFunFEMDetail::mpi_sum(
        IntegrationFunFEMDetail::interface_funfem_integral_local(fh, interface, component, t));
}

template <typename M>
double integral(FunFEM<M> &fh, const Interface<M> &interface, int component) {
    return integral(fh, interface, component, -1.);
}

template <typename M>
double integral(FunFEM<M> &fh, const Interface<M> *interface, int component) {
    return integral(fh, *interface, component, -1.);
}

// -----------------------------------------------------------------------------
// Time-interface integration
// -----------------------------------------------------------------------------

template <typename M>
double integral(FunFEM<M> &fh, const TimeSlab &In, const TimeInterface<M> &gamma,
                int component) {
    return IntegrationFunFEMDetail::mpi_sum(
        IntegrationFunFEMDetail::time_interface_funfem_integral_local(fh, In, gamma, component));
}

template <typename M, typename E>
double integral(const std::shared_ptr<E> &fh, const TimeSlab &In,
                const TimeInterface<M> &gamma, int /*component*/) {
    return IntegrationFunFEMDetail::mpi_sum(
        IntegrationFunFEMDetail::time_interface_expression_integral_local(fh, In, gamma));
}

// -----------------------------------------------------------------------------
// Active mesh boundary integration over selected labels
// -----------------------------------------------------------------------------

template <typename M>
double integral_dK_cut(const std::shared_ptr<const ExpressionVirtual> &fh,
                       const ActiveMesh<M> &Th, const TimeSlab &In,
                       int k, int ifac, const QuadratureFormular1d &qTime, int itq) {
    return IntegrationFunFEMDetail::boundary_cut_face_integral_local(fh, Th, In, k, ifac, qTime, itq);
}

template <typename M>
double integral_dK(const std::shared_ptr<const ExpressionVirtual> &fh,
                   const ActiveMesh<M> &Th, const TimeSlab &In, int k,
                   const typename M::BorderElement &BE, int ifac,
                   const QuadratureFormular1d &qTime, int itq) {
    return IntegrationFunFEMDetail::boundary_uncut_face_integral_local(fh, Th, In, k, BE, ifac, qTime, itq);
}

template <typename M>
double integral(const ActiveMesh<M> &Th, const TimeSlab &In,
                const std::shared_ptr<const ExpressionVirtual> &fh,
                const CBorder & /*b*/, const QuadratureFormular1d &qTime,
                int domain, std::list<int> label) {
    typedef typename M::BorderElement BorderElement;

    const bool all_labels = label.empty();
    double value = 0.;

    for (int itq = 0; itq < qTime.n; ++itq) {
        for (int idx_be = Th.first_boundary_element();
             idx_be < Th.last_boundary_element();
             idx_be += Th.next_boundary_element()) {
            int ifac = -1;
            const int kb = Th.Th.BoundaryElement(idx_be, ifac);
            const BorderElement &BE(Th.be(idx_be));

            if (!all_labels && !util::contain(label, BE.lab))
                continue;

            std::vector<int> idxK = Th.idxAllElementFromBackMesh(kb, -1);
            for (std::size_t i = 0; i < idxK.size(); ++i) {
                const int k = idxK[i];
                if (Th.get_domain_element(k) != domain)
                    continue;
                if (Th.isInactive(k, itq))
                    continue;

                if (Th.isCutFace(k, ifac, itq)) {
                    value += integral_dK_cut(fh, Th, In, k, ifac, qTime, itq);
                } else {
                    value += integral_dK(fh, Th, In, k, BE, ifac, qTime, itq);
                }
            }
        }
    }

    return IntegrationFunFEMDetail::mpi_sum(value);
}

template <typename M>
double integral(const ActiveMesh<M> &Th, const TimeSlab &In,
                const std::shared_ptr<const ExpressionVirtual> &fh,
                const CBorder &b, const QuadratureFormular1d &qTime,
                std::list<int> label = {}) {
    double value = 0.;
    for (int domain = 0; domain < Th.get_nb_domain(); ++domain) {
        value += integral(Th, In, fh, b, qTime, domain, label);
    }
    return value;
}

#include "normsFunFEM.hpp"

#endif // INTEGRATION_FUNFEM_HPP_
