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
CutFEM-Library. If not, see <https://www.gnu.org/licenses/>
*/
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include "../tool.hpp"
#include "../num/matlab.hpp"
#include "../num/gnuplot.hpp"

// #define DARCY_FITTED
#define DARCY_UNFITTED_NEUMANN
// #define DARCY_UNFITTED_3D

#ifdef DARCY_FITTED
    namespace Data_MixedDarcy {
    bool solHasJump = true;
    R d_x           = 1.;
    R d_y           = 1.;
    R shift         = 0.5;
    R interfaceRad  = 0.2501;
    R mu_G          = 2. / 3 * interfaceRad;
    R pie           = M_PI;

    // [u+grad p = 0]
    // R fun_force(double * P, int compInd) {
    //   return 0;
    // }
    // R fun_div(double * P, int compInd) {// is also exact divergence
    //   return -8*pie*pie*sin(2*pie*P[0])*cos(2*pie*P[1]);
    // }
    // R fun_exact_u(double * P, int compInd) {
    //   if (compInd==0) {
    //     return 2*pie*cos(2*pie*P[0])*cos(2*pie*P[1]);
    //   } else {
    //     return -2*pie*sin(2*pie*P[0])*sin(2*pie*P[1]);
    //   }
    // }
    // R fun_exact_p(double * P, int compInd) {
    //   return -sin(2*pie*P[0])*cos(2*pie*P[1]);
    // }

    // // [divu = 0]
    // R fun_force(double *P, int compInd) { return 0; }
    // R fun_div(double *P, int compInd) { // is also exact divergence
    //    return 0;
    // }
    // R fun_exact_u(double *P, int compInd) {
    //    if (compInd == 0) {
    //       return cos(P[0]) * sinh(P[1]);
    //    } else {
    //       return sin(P[0]) * cosh(P[1]);
    //    }
    // }
    // R fun_exact_p(double *P, int compInd) {
    //    return -sin(P[0]) * sinh(P[1]) - (cos(1) - 1) * (cosh(1) - 1);
    // }

    // [divu = 0, u linear, DOMAIN = [0,1]^2]
    R fun_force(double *P, int compInd) { return 0; }
    R fun_div(double *P, int compInd) {
        R x = P[0];
        R y = P[1];
        return 0;
    }
    R fun_exact_u(double *P, int compInd) {
        R x = P[0];
        R y = P[1];
        if (compInd == 0)
            return -x;
        else
            return y - 1;
    }
    R fun_exact_p(double *P, int compInd) {
        R x = P[0];
        R y = P[1];
        return -(-x * x / 2 + (y - 1) * (y - 1) / 2);
    }

    } // namespace Data_MixedDarcy
    using namespace Data_MixedDarcy;

    int main(int argc, char **argv) {
        typedef TestFunction<2> FunTest;
        typedef FunFEM<Mesh2> Fun_h;
        typedef Mesh2 Mesh;
        typedef ActiveMeshT2 CutMesh;
        typedef FESpace2 Space;
        typedef CutFESpaceT2 CutSpace;

        // MPIcf cfMPI(argc, argv);
        const double cpubegin   = CPUtime();
        globalVariable::verbose = 1;

        int nx = 21; // 6
        int ny = 21; // 6
        // Kh_init0.truncate(interface_init, -1);

        std::vector<double> uPrint, pPrint, divPrint, divPrintLoc, maxDivPrint, h, convuPr, convpPr, convdivPr,
            convdivPrLoc, convmaxdivPr;

        int iters = 1;
        for (int i = 0; i < iters; ++i) {
            Mesh Kh(nx, ny, 0., 0., d_x, d_y);
            // Mesh2 Kh("../FittedMesh.msh");

            Space Vh(Kh, DataFE<Mesh>::BDM2);
            Space Qh(Kh, DataFE<Mesh>::P1dc);
            Space Q2h(Kh, DataFE<Mesh>::P2dc);
            Lagrange2 FEvelocity(2);
            Space V2h(Kh, FEvelocity);

            FEM<Mesh2> darcy(Vh);
            darcy.add(Qh);
            const R h_i  = 1. / (nx - 1);
            const R invh = 1. / h_i;

            Normal n;
            Tangent t;

            Fun_h fv(Vh, fun_force);
            Fun_h fq(Q2h, fun_div);
            Fun_h p0(Q2h, fun_exact_p);
            Fun_h u0(Vh, fun_exact_u);
            Fun_h uu0(V2h, fun_exact_u);

            FunTest p(Qh, 1), q(Qh, 1), u(Vh, 2), v(Vh, 2);

            double penParam = 1e0;

            // [ASSEMBLY]
            darcy.addBilinear(innerProduct(u, v) - innerProduct(p, div(v)) + innerProduct(div(u), q), Kh);
            darcy.addLinear(innerProduct(fq.expr(), q), Kh);

            // [Essential conditions]
            // darcy.addBilinear(
            //   // +innerProduct(p, v*n) // [can be removed in the fitted case!]
            //   +innerProduct(penParam*u*n, invh*v*n)
            //   , Kh, INTEGRAL_BOUNDARY
            // );
            // darcy.addLinear(
            //   +innerProduct(u0*n, penParam*invh*v*n)
            //   , Kh, INTEGRAL_BOUNDARY
            // );

            // darcy.setDirichlet(u0, Khi);

            // R meanP = integral(Kh, exactp, 0);
            // darcy.addLagrangeMultiplier(innerProduct(1., p), 0, Kh);
            // int N = Vh.get_nb_dof() + Qh.get_nb_dof();
            // darcy.mat_[std::make_pair(N,N)] = -1;

            // [Natural conditions]
            darcy.addLinear(-innerProduct(p0.expr(), v * n), Kh, INTEGRAL_BOUNDARY);

            // matlab::Export(darcy.mat_, "mat"+to_string(i)+".dat");
            darcy.solve("umfpack");

            // std::cout << darcy.rhs_(N) << std::endl;

            // EXTRACT SOLUTION
            int idx0_s  = Vh.get_nb_dof();
            Rn_ data_uh = darcy.rhs_(SubArray(Vh.get_nb_dof(), 0));
            Rn_ data_ph = darcy.rhs_(SubArray(Qh.get_nb_dof(), idx0_s));
            Fun_h uh(Vh, data_uh);
            Fun_h ph(Qh, data_ph);
            auto femSol_0dx = dx(uh.expr(0));
            auto femSol_1dy = dy(uh.expr(1));

            // L2 norm vel
            R errU      = L2norm(uh, fun_exact_u, 0, 2);
            R errP      = L2norm(ph, fun_exact_p, 0, 1);
            R errDiv    = L2norm(femSol_0dx + femSol_1dy, fun_div, Kh);
            R maxErrDiv = maxNorm(femSol_0dx + femSol_1dy, Kh);

            // [PLOTTING]
            {
                // Fun_h soluh(Vh, fun_exact_u);
                // Fun_h solph(Vh, fun_exact_p);
                // Fun_h solh(Wh, fun_exact);
                // solh.v -= uh.v;
                // solh.v.map(fabs);
                // Fun_h divSolh(Vh, fun_div);
                // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);
                Paraview<Mesh> writer(Kh, "darcyBDM2_" + std::to_string(i) + ".vtk");
                writer.add(uh, "velocity", 0, 2);
                writer.add(u0, "velocityExact", 0, 2);
                writer.add(ph, "pressure", 0, 1);
                writer.add(p0, "pressureExact", 0, 1);
                writer.add(femSol_0dx + femSol_1dy, "divergence");
                writer.add(fabs((femSol_0dx + femSol_1dy) - fq.expr()), "divergenceError");
            }

            pPrint.push_back(errP);
            uPrint.push_back(errU);
            // continue;
            divPrint.push_back(errDiv);
            // divPrintLoc.push_back(errDivLoc);

            maxDivPrint.push_back(maxErrDiv);
            h.push_back(h_i);

            if (i == 0) {
                convpPr.push_back(0);
                convuPr.push_back(0);
                convdivPr.push_back(0);
                convdivPrLoc.push_back(0);
                convmaxdivPr.push_back(0);
            } else {
                convpPr.push_back(log(pPrint[i] / pPrint[i - 1]) / log(h[i] / h[i - 1]));
                convuPr.push_back(log(uPrint[i] / uPrint[i - 1]) / log(h[i] / h[i - 1]));
                convdivPr.push_back(log(divPrint[i] / divPrint[i - 1]) / log(h[i] / h[i - 1]));
                // convdivPrLoc.push_back(
                // log(divPrintLoc[i]/divPrintLoc[i-1])/log(h[i]/h[i-1]));

                convmaxdivPr.push_back(log(maxDivPrint[i] / maxDivPrint[i - 1]) / log(h[i] / h[i - 1]));
            }

            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            // nx += 1;
            // ny += 1;
            // nx = (int)round( (1+0.2*i)*nx/2 )*2; // Makes a nonuniform refinement
            // to an EVEN integer ny = (int)round( (1+0.2*i)*ny/2 )*2; std::cout << nx
            // << std::endl; shift = 0.5+(i+1)*h_i/iters; // moves one grid cell over
            // entire span
        }
        std::cout << "\n"
                << std::left << std::setw(10) << std::setfill(' ') << "h" << std::setw(15) << std::setfill(' ') << "err_p"
                << std::setw(15) << std::setfill(' ') << "conv p" << std::setw(15) << std::setfill(' ') << "err u"
                << std::setw(15) << std::setfill(' ') << "conv u" << std::setw(15) << std::setfill(' ')
                << "err divu"
                // << std::setw(15) << std::setfill(' ') << "conv divu"
                // << std::setw(15) << std::setfill(' ') << "err_new divu"
                // << std::setw(15) << std::setfill(' ') << "convLoc divu"
                << std::setw(15) << std::setfill(' ')
                << "err maxdivu"
                // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
                << "\n"
                << std::endl;
        for (int i = 0; i < uPrint.size(); ++i) {
            std::cout << std::left << std::setw(10) << std::setfill(' ') << h[i] << std::setw(15) << std::setfill(' ')
                    << pPrint[i] << std::setw(15) << std::setfill(' ') << convpPr[i] << std::setw(15) << std::setfill(' ')
                    << uPrint[i] << std::setw(15) << std::setfill(' ') << convuPr[i] << std::setw(15) << std::setfill(' ')
                    << divPrint[i]
                    // << std::setw(15) << std::setfill(' ') << convdivPr[i]
                    // << std::setw(15) << std::setfill(' ') << divPrintLoc[i]
                    // << std::setw(15) << std::setfill(' ') << convdivPrLoc[i]
                    << std::setw(15) << std::setfill(' ')
                    << maxDivPrint[i]
                    // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
                    << std::endl;
        }
    }

#endif

#ifdef DARCY_UNFITTED_NEUMANN
    // ================== Namespace: Quarter Circle ==================
    namespace Data_quarter_circle { // u - grad p = 0
        constexpr bool hybrid_fitted_unfitted = true;
        constexpr double d_x = 2.0;
        constexpr double d_y = 2.0;
        constexpr double shift = 0.0;
        constexpr double interfaceRad = 0.501;

        R fun_radius2(double* P) {
            return (P[0] - shift) * (P[0] - shift) + (P[1] - shift) * (P[1] - shift);
        }

        R fun_levelSet(double* P, int) {
            return std::sqrt(fun_radius2(P)) - interfaceRad;
        }

        R fun_natural(double* P, int) {
            return std::sin(P[0]) * std::sinh(P[1]) + (std::cos(1) - 1) * (std::cosh(1) - 1);
        }

        R fun_enforced(double* P, int compInd) {
            return (compInd == 0) ? std::cos(P[0]) * std::sinh(P[1]) : std::sin(P[0]) * std::cosh(P[1]);
        }

        R fun_force(double*, int) { return 0.0; }

        R fun_div(double*, int, int) { return 0.0; }

        R fun_exact_u(double* P, int compInd, int) {
            return (compInd == 0) ? std::cos(P[0]) * std::sinh(P[1]) : std::sin(P[0]) * std::cosh(P[1]);
        }

        R fun_exact_p(double* P, int, int) {
            return std::sin(P[0]) * std::sinh(P[1]) + (std::cos(1) - 1) * (std::cosh(1) - 1);
        }
    }

    // ================== Namespace: Circle ==================
    namespace Data_circle {
        constexpr bool hybrid_fitted_unfitted = false;
        constexpr double d_x = 1.0;
        constexpr double d_y = 1.0;
        constexpr double shift = 0.5;
        constexpr double interfaceRad = 0.4501; 

        R fun_levelSet(double* P, int) {
            return interfaceRad - std::sqrt((P[0] - shift) * (P[0] - shift) + (P[1] - shift) * (P[1] - shift));
        }

        R fun_natural(double* P, int) {
            return std::sin(2 * M_PI * P[0]) * std::cos(2 * M_PI * P[1]);
        }

        R fun_enforced(double* P, int compInd) {
            return (compInd == 0) ? 2 * M_PI * std::cos(2 * M_PI * P[0]) * std::cos(2 * M_PI * P[1]) :
                (compInd == 1) ? -2 * M_PI * std::sin(2 * M_PI * P[0]) * std::sin(2 * M_PI * P[1]) :
                                    0.0;
        }

        R fun_force(double*, int) { return 0.0; }

        R fun_div(double* P, int, int) {
            return -8 * M_PI * M_PI * std::sin(2 * M_PI * P[0]) * std::cos(2 * M_PI * P[1]);
        }

        R fun_exact_u(double* P, int compInd, int) {
            return (compInd == 0) ? 2 * M_PI * std::cos(2 * M_PI * P[0]) * std::cos(2 * M_PI * P[1]) :
                                -2 * M_PI * std::sin(2 * M_PI * P[0]) * std::sin(2 * M_PI * P[1]);
        }

        R fun_exact_p(double* P, int, int) {
            return std::sin(2 * M_PI * P[0]) * std::cos(2 * M_PI * P[1]);
        }
    }

    // using namespace Data_quarter_circle;
    using namespace Data_circle;

    // ================== Main Function ==================
    int main(int argc, char **argv) {
        using Mesh = Mesh2;
        using FunTest = TestFunction<Mesh2>;
        using Fun_h = FunFEM<Mesh2>;
        using CutMesh = ActiveMeshT2;
        using Space = FESpace2;
        using CutSpace = CutFESpaceT2;

        // Timing
        const double cpubegin = CPUtime();

        // Mesh resolution
        int numCellsX = 11, numCellsY = 11;
        int iters = 4;

        // Data storage
        std::vector<double> velocityErrors, pressureErrors, divergenceErrors, maxDivergenceErrors, meshSizes;
        std::vector<double> convuPr, convpPr, convdivPr, convmaxdivPr;

        for (int i = 0; i < iters; ++i) {
            std::cout << "Iteration " << i << std::endl;
            Mesh mesh(numCellsX, numCellsY, 0., 0., d_x, d_y);
            
            // Function spaces
            Space Lh(mesh, DataFE<Mesh2>::P1);
            Space Vh(mesh, DataFE<Mesh>::RT0);
            Space Qh(mesh, DataFE<Mesh>::P0);
            Space Q2h(mesh, DataFE<Mesh>::P2); // for RHS

            // Level set and interface
            Fun_h levelSet(Lh, fun_levelSet);
            InterfaceLevelSet<Mesh> interface(mesh, levelSet);

            // Cut FEM setup
            ActiveMesh<Mesh> cutMesh(mesh);
            cutMesh.truncate(interface, -1);
            MacroElement<Mesh> macro(cutMesh, 0.25);

            // GNUPLOT
            gnuplot::save(mesh);
            gnuplot::save(interface);
            // gnuplot::save(macro, extension);
            // gnuplot::save(macroInterface);
            gnuplot::save(cutMesh, "Thi.dat");
            break;

            CutSpace Wh(cutMesh, Vh);
            CutSpace Ph(cutMesh, Qh);
            CutSpace P2h(cutMesh, Q2h);

            CutFEM<Mesh2> darcySolver(Wh);
            darcySolver.add(Ph);

            double h = 1.0 / (numCellsX - 1);
            double invh = 1.0 / h;

            // Define functions on cut spaces
            Fun_h fv(Wh, fun_force);
            Fun_h fq(P2h, fun_div);
            Fun_h u0(Wh, fun_enforced);
            Fun_h p0(P2h, fun_natural);

            Normal n;
            Tangent t;
            FunTest p(Ph, 1), q(Ph, 1), u(Wh, 2), v(Wh, 2);

            // Penalty parameters
            double velocityPenalty = 1e0;
            double pressurePenalty = 1e0;
            double generalPenalty = 1e0;

            // Assembly
            darcySolver.addBilinear(
                innerProduct(u, v)  
                +innerProduct(p, div(v))  
                +innerProduct(div(u), q)
            , cutMesh);
            darcySolver.addLinear(
                innerProduct(fq.exprList(), q)
            , cutMesh);

            // Face stabilization
            FunTest grad2un = grad(grad(u) * n) * n;
            darcySolver.addFaceStabilization( // [h^(2k+1) h^(2k+1)]
                innerProduct(velocityPenalty*h*jump(u), jump(v))
                // +innerProduct(velocityPenalty*pow(h,3)*jump(grad(u)*n), jump(grad(v)*n))
                // +innerProduct(uPenParam*pow(h,5)*jump(grad2un), jump(grad2un))
                +innerProduct(pressurePenalty*h*jump(p), jump(div(v)))
                +innerProduct(pressurePenalty*h*jump(div(u)), jump(q))
                // -innerProduct(pPenParam*pow(h,3)*jump(grad(p)), jump(grad(div(v))))
                // +innerProduct(pPenParam*pow(h,3)*jump(grad(div(v))) ,
                // jump(grad(q))) 
            , cutMesh, macro); 

            // Natural conditions
            darcySolver.addLinear(
                innerProduct(p0.exprList(), v * n)
            , interface);
            if (hybrid_fitted_unfitted) {
                darcySolver.addLinear(
                    innerProduct(p0.exprList(), v * n)
                // , mesh, INTEGRAL_BOUNDARY);
                , cutMesh, INTEGRAL_BOUNDARY);
            }

            // Essential conditions
            // darcySolver.addLagrangeMultiplier(
            //     innerProduct(1.0, p)
            //     , 0.0, cutMesh);
            // darcySolver.addBilinear(
            //     innerProduct(p, v * n) 
            //     +innerProduct(generalPenalty * u * n, invh * v * n)
            //     , interface);
            // darcySolver.addLinear(
            //     innerProduct(u0 * n, generalPenalty * invh * v * n)
            //     , interface);

            // Export matrix
            matlab::Export(darcySolver.mat_[0], "mat" + std::to_string(i) + "Cut.dat");
            // Solve system
            darcySolver.solve("umfpack");

            // Extract solution
            Rn_ data_uh = darcySolver.rhs_(SubArray(Wh.get_nb_dof(), 0));
            Rn_ data_ph = darcySolver.rhs_(SubArray(Ph.get_nb_dof(), Wh.get_nb_dof()));
            Fun_h uh(Wh, data_uh);
            Fun_h ph(Ph, data_ph);
            
            // [Paraview]
            {
                Fun_h solu(Wh, fun_exact_u);
                Fun_h soluErr(Wh, fun_exact_u);
                soluErr.v -= uh.v;
                soluErr.v.map(fabs);
                Fun_h solp(Ph, fun_exact_p);

                // Fun_h divSolh(Wh, fun_div);
                // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

                Paraview<Mesh> writer(cutMesh, "darcy_" + std::to_string(i) + ".vtk");

                writer.add(uh, "velocity", 0, 2);
                writer.add(ph, "pressure", 0, 1);

                // writer.add(dx_uh0+dy_uh1, "divergence");
                // writer.add(femSol_0dx+femSol_1dy+fflambdah, "divergence");

                writer.add(solu, "velocityExact", 0, 2);
                writer.add(soluErr, "velocityError", 0, 2);
                writer.add(solp, "pressureExact", 0, 1);
            }

            double errU = L2normCut(uh, fun_exact_u, 0, 2);
            double errP = L2normCut(ph, fun_exact_p, 0, 1);
            double errDiv = L2normCut(dx(uh.expr(0)) + dy(uh.expr(1)), fun_div, cutMesh);
            double maxErrDiv = maxNormCut(dx(uh.expr(0)) + dy(uh.expr(1)), fun_div, cutMesh);

            // Store results
            pressureErrors.push_back(errP);
            velocityErrors.push_back(errU);
            divergenceErrors.push_back(errDiv);
            maxDivergenceErrors.push_back(maxErrDiv);
            meshSizes.push_back(h);

            if (i == 0) {
                convpPr.push_back(0);
                convuPr.push_back(0);
                convdivPr.push_back(0);
                convmaxdivPr.push_back(0);
            } else {
                convpPr.push_back(log(pressureErrors[i] / pressureErrors[i - 1]) / log(meshSizes[i] / meshSizes[i - 1]));
                convuPr.push_back(log(velocityErrors[i] / velocityErrors[i - 1]) / log(meshSizes[i] / meshSizes[i - 1]));
                convdivPr.push_back(log(divergenceErrors[i] / divergenceErrors[i - 1]) / log(meshSizes[i] / meshSizes[i - 1]));
                convmaxdivPr.push_back(log(maxDivergenceErrors[i] / maxDivergenceErrors[i - 1]) / log(meshSizes[i] / meshSizes[i - 1]));
            }

            // Refine mesh
            numCellsX = 2 * numCellsX - 1;
            numCellsY = 2 * numCellsY - 1;
        }

        // Output results
        std::cout << "\n" << std::left << std::setw(10) << "h"
            << std::setw(15) << "err_p"
            << std::setw(15) << "conv_p"
            << std::setw(15) << "err_u"
            << std::setw(15) << "conv_u"
            << std::setw(15) << "err_divu"
            << std::setw(15) << "err_maxdivu" << "\n";

        for (size_t i = 0; i < velocityErrors.size(); ++i) {
            std::cout << std::left << std::setw(10) << meshSizes[i]
                << std::setw(15) << pressureErrors[i]
                << std::setw(15) << convpPr[i]
                << std::setw(15) << velocityErrors[i]
                << std::setw(15) << convuPr[i]
                << std::setw(15) << divergenceErrors[i]
                << std::setw(15) << maxDivergenceErrors[i] << "\n";
        }

        return 0;
    }


#endif



#ifdef DARCY_UNFITTED_3D

    R shift        = 0.;                    // 0.5;
    R interfaceRad = 1.;                    // 0.250001; // not exactly 1/4 to avoid interface cutting
                                            // exaclty a vertex
    R mu_G         = 2. / 3 * interfaceRad; // xi0*mu_G = 1/8*2/3*1/4

    R fun_radius2(const R3 P) {
        return (P[0] - shift) * (P[0] - shift) + (P[1] - shift) * (P[1] - shift) + (P.z - shift) * (P.z - shift);
    }
    R fun_levelSet(const R3 P, const int i) {
        return sqrt((P[0] - shift) * (P[0] - shift) + (P[1] - shift) * (P[1] - shift) + (P.z - shift) * (P.z - shift)) -
            interfaceRad;
    }

    R fun_dirichlet(const R3 P, int compInd) { return 0; }
    R fun_neumann(const R3 P, int compInd, int dom) {
        R r2      = fun_radius2(P);
        R radius2 = interfaceRad * interfaceRad;
        // return r2/(2*radius2)+3./2.;
        return r2 / (1 * radius2);
        // return r2 - radius2;//
        // return 1.;
    }

    R fun_force(const R3 P, int compInd) { return 0; }
    R fun_div(const R3 P, int compInd, int dom) { // is also exact divergence
        R radius2 = interfaceRad * interfaceRad;
        // if (dom==0) // r2>radius2
        // return -2./radius2;
        // else
        return -4. / radius2;
    }
    R fun_exact_u(const R3 P, int compInd, int dom) {
        Diff<R, 3> X(P[0], 0), Y(P[1], 1), Z(P.z, 2);
        Diff<R, 3> r2  = (X - shift) * (X - shift) + (Y - shift) * (Y - shift) + (Z - shift) * (Z - shift);
        R radius2      = interfaceRad * interfaceRad;
        R cst          = 0;  //(dom==0)*3./2;
        R mul          = 1.; //(dom==0)*2 + (dom==1)*1;
        Diff<R, 3> val = r2 / (mul * radius2) + cst;
        return -val.d[compInd];

        // return P.norme();
    }
    R fun_exact_p(const R3 P, int compInd, int dom) {
        Diff<R, 3> X(P[0], 0), Y(P[1], 1), Z(P.z, 2);
        Diff<R, 3> r2  = (X - shift) * (X - shift) + (Y - shift) * (Y - shift) + (Z - shift) * (Z - shift);
        R radius2      = interfaceRad * interfaceRad;
        R cst          = 0;  //(dom==0)*3./2;
        R mul          = 1.; //(dom==0)*2 + (dom==1)*1;
        Diff<R, 3> val = r2 / (mul * radius2) + cst;
        return val.val;
    }
    R fun_interfacePr(const R3 P, int compInd) { return 19. / 12; }

    int main(int argc, char **argv) {
        typedef TestFunction<3> FunTest;
        typedef Mesh3 Mesh;
        typedef FunFEM<Mesh> Fun_h;
        typedef ActiveMesh<Mesh> CutMesh;
        typedef GFESpace<Mesh> Space;
        typedef CutFESpace<Mesh> CutSpace;

        MPIcf cfMPI(argc, argv);
        const double cpubegin = CPUtime();

        int nx = 21;

        std::vector<double> uPrint, pPrint, divPrint, divPrintLoc, maxDivPrint, h, convuPr, convpPr, convdivPr,
            convdivPrLoc, convmaxdivPr;
        std::vector<double> ratioCut1, ratioCut2;
        int iters = 2;

        for (int i = 0; i < iters; ++i) {
            // Mesh Kh(nx, nx, nx, 0., 0., 0., 1., 1.,1.);
            double ox = -1.27;
            double lx = 2 * fabs(ox);
            Mesh Kh(nx, nx, nx, ox, ox, ox, lx, lx, lx);

            Kh.info();

            Space Lh(Kh, DataFE<Mesh>::P1);
            Fun_h levelSet(Lh, fun_levelSet);
            InterfaceLevelSet<Mesh> interface(Kh, levelSet);

            Lagrange3 FEvelocity(2);
            Space Vh(Kh, DataFE<Mesh>::RT0);
            Space Qh(Kh, DataFE<Mesh>::P0);

            // ActiveMesh<Mesh> Kh_i(Kh, interface);
            ActiveMesh<Mesh> Kh_i(Kh);
            Kh_i.truncate(interface, 1);
            Kh_i.info();

            CutSpace Wh(Kh_i, Vh);
            Wh.info();
            CutSpace Ph(Kh_i, Qh);
            Ph.info();

            CutFEM<Mesh> darcy(Wh);
            darcy.add(Ph);
            const R h_i  = 1. / (nx - 1);
            const R invh = 1. / h_i;
            // MacroElement<Mesh> macro(Kh_i, 0.25);
            R xi         = 3. / 4;
            R xi0        = (xi - 0.5) / 2.;

            // We define fh on the cutSpace
            Fun_h fq(Ph, fun_div);
            Fun_h p0(Lh, fun_neumann);
            Fun_h phat(Ph, fun_interfacePr);
            // Fun_h pex(Ph, fun_exact_p); ExpressionFunFEM<Mesh> exactp(pex,0,op_id);
            Fun_h u0(Wh, fun_exact_u);

            // std::cout << integral(p0, interface, 0) << std::endl;
            // nx = 2*nx -1;
            // continue;

            Normal n;
            Tangent t;
            FunTest p(Ph, 1), q(Ph, 1), u(Wh, 3), v(Wh, 3);

            double uPenParam = 1e0; // 1e-2; //cont 1e-1`
            double pPenParam = 1e0; // 1e1; // cont 1e2
            double jumpParam = 1e0; // [anything<1e0 doesn't give full u convergence]
            double penParam  = 1e0; // [anything<1e0 doesn't give full u convergence]

            double t0 = MPIcf::Wtime();
            darcy.addBilinear(innerProduct(u, v) - innerProduct(p, div(v)) + innerProduct(div(u), q), Kh_i);
            darcy.addLinear(-innerProduct(p0.exprList(), v * n), interface);
            darcy.addLinear(innerProduct(fq.exprList(), q), Kh_i);

            // darcy.addBilinear(
            //   +innerProduct(p, v*n)
            //   +innerProduct(penParam*u*n, 1./h_i/h_i*v*n)
            //   ,interface
            // );
            // darcy.addLinear(
            //   +innerProduct(u0*n, penParam*1./h_i/h_i*v*n)
            //   ,interface
            // );

            // Fun_h exactph(Ph, fun_exact_p); ExpressionFunFEM<Mesh>
            // exactp(exactph,0,op_id); R meanP = integral(Kh_i,exactp,0);
            // darcy.addLagrangeMultiplier(
            //   innerProduct(1.,p), 0 , Kh_i
            // );

            // t0 = MPIcf::Wtime();

            FunTest grad2un = grad(grad(u) * n) * n;
            darcy.addFaceStabilization( // [h^(2k+1) h^(2k+1)]
                innerProduct(uPenParam * pow(h_i, 1) * jump(u),
                            jump(v)) // [Method 1: Remove jump in vel]
                    + innerProduct(uPenParam * pow(h_i, 3) * jump(grad(u) * n), jump(grad(v) * n))
                    // +innerProduct(uPenParam*pow(h_i,5)*jump(grad2un),
                    // jump(grad2un)) +innerProduct(pPenParam*pow(h_i,1)*jump(p),
                    // jump(q)) +innerProduct(pPenParam*pow(h_i,3)*jump(grad(p)),
                    // jump(grad(q)))

                    //  innerProduct(uPenParam*h_i*jump(u), jump(v)) // [Method 1:
                    //  Remove jump in vel]
                    // +innerProduct(uPenParam*pow(h_i,3)*jump(grad(u)*n),
                    // jump(grad(v)*n))
                    // +innerProduct(uPenParam*pow(h_i,5)*jump(grad2un),
                    // jump(grad2un))
                    - innerProduct(pPenParam * h_i * jump(p), jump(div(v))) +
                    innerProduct(pPenParam * h_i * jump(div(u)), jump(q))
                // -innerProduct(pPenParam*pow(h_i,3)*jump(grad(p)),
                // jump(grad(div(v))))
                // +innerProduct(pPenParam*pow(h_i,3)*jump(grad(div(v))) ,
                // jump(grad(q)))
                ,
                Kh_i
                // , macro
            );

            std::cout << " Time assembly \t" << MPIcf::Wtime() - t0 << std::endl;
            t0 = MPIcf::Wtime();

            darcy.solve();
            std::cout << " Time solver \t" << MPIcf::Wtime() - t0 << std::endl;

            // EXTRACT SOLUTION
            int idx0_s  = Wh.get_nb_dof();
            Rn_ data_uh = darcy.rhs_(SubArray(Wh.get_nb_dof(), 0));
            Rn_ data_ph = darcy.rhs_(SubArray(Ph.get_nb_dof(), idx0_s));
            Fun_h uh(Wh, data_uh);
            Fun_h ph(Ph, data_ph);
            ExpressionFunFEM<Mesh> femSol_0dx(uh, 0, op_dx);
            ExpressionFunFEM<Mesh> femSol_1dy(uh, 1, op_dy);
            ExpressionFunFEM<Mesh> femSol_1dz(uh, 2, op_dz);

            // L2 norm vel
            R errU      = L2normCut(uh, fun_exact_u, 0, 3);
            R errP      = L2normCut(ph, fun_exact_p, 0, 1);
            R errDiv    = L2normCut(femSol_0dx + femSol_1dy + femSol_1dz, fun_div, Kh_i);
            R maxErrDiv = maxNormCut(femSol_0dx + femSol_1dy + femSol_1dz, fun_div, Kh_i);

            // [PLOTTING]
            if (MPIcf::IamMaster()) {
                Fun_h solh(Wh, fun_exact_u);
                Fun_h solph(Ph, fun_exact_p);
                // solh.v -= uh.v;
                // solh.v.map(fabs);
                Fun_h divSolh(Ph, fun_div);
                ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

                // Paraview<Mesh> writer(Kh_i, "darcyRT2_"+to_string(i)+".vtk");
                Paraview<Mesh> writer(Kh_i, "darcyFictitious3D_" + to_string(i) + ".vtk");

                writer.add(uh, "velocity", 0, 3);
                writer.add(solh, "velocityExact", 0, 3);
                writer.add(ph, "pressure", 0, 1);
                writer.add(solph, "pressureExact", 0, 1);
                writer.add(femSol_0dx + femSol_1dy, "divergence");
                writer.add(fabs((femSol_0dx + femSol_1dy + femSol_1dz) - femDiv), "divergenceError");
            }

            pPrint.push_back(errP);
            uPrint.push_back(errU);
            divPrint.push_back(errDiv);
            maxDivPrint.push_back(maxErrDiv);
            h.push_back(h_i);

            if (i == 0) {
                convpPr.push_back(0);
                convuPr.push_back(0);
                convdivPr.push_back(0);
                convdivPrLoc.push_back(0);
                convmaxdivPr.push_back(0);
            } else {
                convpPr.push_back(log(pPrint[i] / pPrint[i - 1]) / log(h[i] / h[i - 1]));
                convuPr.push_back(log(uPrint[i] / uPrint[i - 1]) / log(h[i] / h[i - 1]));
                convdivPr.push_back(log(divPrint[i] / divPrint[i - 1]) / log(h[i] / h[i - 1]));
                convmaxdivPr.push_back(log(maxDivPrint[i] / maxDivPrint[i - 1]) / log(h[i] / h[i - 1]));
            }
            nx = 2 * nx - 1;
        }

        std::cout << "\n"
                << std::left << std::setw(10) << std::setfill(' ') << "h" << std::setw(15) << std::setfill(' ')
                << "err_p"
                // << std::setw(15) << std::setfill(' ') << "conv p"
                << std::setw(15) << std::setfill(' ')
                << "err u"
                // << std::setw(15) << std::setfill(' ') << "conv u"
                << std::setw(15) << std::setfill(' ')
                << "err divu"
                // << std::setw(15) << std::setfill(' ') << "conv divu"
                // << std::setw(15) << std::setfill(' ') << "err_new divu"
                // << std::setw(15) << std::setfill(' ') << "convLoc divu"
                << std::setw(15) << std::setfill(' ')
                << "err maxdivu"
                // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
                << "\n"
                << std::endl;
        for (int i = 0; i < uPrint.size(); ++i) {
            std::cout << std::left << std::setprecision(5) << std::setw(10) << std::setfill(' ') << h[i] << std::setw(15)
                    << std::setfill(' ')
                    << pPrint[i]
                    // << std::setw(15) << std::setfill(' ') << convpPr[i]
                    << std::setw(15) << std::setfill(' ')
                    << uPrint[i]
                    // << std::setw(15) << std::setfill(' ') << convuPr[i]
                    << std::setw(15) << std::setfill(' ')
                    << divPrint[i]
                    // << std::setw(15) << std::setfill(' ') << convdivPr[i]
                    // << std::setw(15) << std::setfill(' ') << divPrintLoc[i]
                    // << std::setw(15) << std::setfill(' ') << convdivPrLoc[i]
                    << std::setw(15) << std::setfill(' ')
                    << maxDivPrint[i]
                    // << std::setw(15) << std::setfill(' ') << ratioCut1[i]
                    // << std::setw(15) << std::setfill(' ') << ratioCut2[i]
                    // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
                    << std::endl;
        }
    }

#endif