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

#define TORUS_UNFITTED_3D


#ifdef TORUS_UNFITTED_3D

    namespace Data_Sphere {
        R major_rad = 0.5; 
        R rad = 0.25;                   

        R fun_major_rad_sq(double *P) {
            return P[0]*P[0] + P[1]*P[1];
        }

        R fun_levelSet(double *P, const int i) {
            return sqrt(P[0] * P[0] + P[1] * P[1] + P[2] * P[2]) - 0.75;
        }

        R fun_zero(double *P, int comp, int dom) {
            return 0;
        }

        R fun_force(double *P, int comp, int dom) { 
            R x = P[0];
            R y = P[1];
            R r = sqrt(fun_major_rad_sq(P));
            if (comp == 0) {
                return -3 * x * y;// / pow(r, 5);
            } else if (comp == 1) {
                return (x * x - 2 * y * y);// / pow(r, 5);
            } else if (comp == 2) {
                return 0;
            }
            return 0;
        }
        R fun_force_x(double *P, int comp, int dom) { 
            return fun_force(P, 0, dom);
        }
        R fun_force_y(double *P, int comp, int dom) { 
            return fun_force(P, 1, dom);
        }

        R fun_exact_u(double *P, int comp, int dom) {
            R x = P[0];
            R y = P[1];
            R r = sqrt(fun_major_rad_sq(P));
            if (comp == 0) {
                return -x * y;// / pow(r, 3);
            } else if (comp == 1) {
                return x * x;// / pow(r, 3);
            } else if (comp == 2) {
                return 0;
            }
            return 0;
        }
        
        R fun_div_u(double *P, int comp, int dom) { // is also exact divergence
            R x = P[0];
            R y = P[1];
            R r = sqrt(fun_major_rad_sq(P));
            return -y;// / pow(r,3);
        }

        R fun_exact_w(double *P, int comp, int dom) { // is also exact divergence
            return fun_div_u(P, comp, dom);
        }

        R fun_closed_form(double *P, int comp, int dom) {
            R x = P[0];
            R y = P[1];
            R r = sqrt(fun_major_rad_sq(P));
            if (comp == 0) {
                return -y;// / pow(r,2);
            } else if (comp == 1) {
                return x;// / pow(r,2);
            } else if (comp == 2) {
                return 0;
            }
            return 0;
        }
    }

    namespace Data_Torus {
        R major_rad = 0.5; 
        R rad = 0.25;                   

        R fun_major_rad_sq(double *P) {
            return P[0]*P[0] + P[1]*P[1];
        }

        R fun_levelSet(double *P, const int i) {
            return sqrt( pow(sqrt(fun_major_rad_sq(P)) - major_rad, 2) + P[2]*P[2] ) - rad;
        }

        R fun_zero(double *P, int comp, int dom) {
            return 0;
        }

        R fun_force(double *P, int comp, int dom) { 
            R x = P[0];
            R y = P[1];
            R r = sqrt(fun_major_rad_sq(P));
            if (comp == 0) {
                return -3 * x * y / pow(r, 5);
            } else if (comp == 1) {
                return (x * x - 2 * y * y) / pow(r, 5);
            } else if (comp == 2) {
                return 0;
            }
            return 0;
        }
        R fun_force_x(double *P, int comp, int dom) { 
            return fun_force(P, 0, dom);
        }
        R fun_force_y(double *P, int comp, int dom) { 
            return fun_force(P, 1, dom);
        }

        R fun_exact_u(double *P, int comp, int dom) {
            R x = P[0];
            R y = P[1];
            R r = sqrt(fun_major_rad_sq(P));
            if (comp == 0) {
                return -x * y / pow(r, 3);
            } else if (comp == 1) {
                return x * x / pow(r, 3);
            } else if (comp == 2) {
                return 0;
            }
            return 0;
        }
        
        R fun_div_u(double *P, int comp, int dom) { // is also exact divergence
            R x = P[0];
            R y = P[1];
            R r = sqrt(fun_major_rad_sq(P));
            return -y / pow(r,3);
        }

        R fun_exact_w(double *P, int comp, int dom) { // is - exact divergence
            return -fun_div_u(P, comp, dom);
        }

        R fun_closed_form(double *P, int comp, int dom) { // harmonic form, with -div and curl = 0
            R x = P[0];
            R y = P[1];
            R r = sqrt(fun_major_rad_sq(P));
            if (comp == 0) {
                return -y / pow(r,2);
            } else if (comp == 1) {
                return x / pow(r,2);
            } else if (comp == 2) {
                return 0;
            }
            return 0;
        }
    }

    // using namespace Data_Sphere;
    using namespace Data_Torus;
    int main(int argc, char **argv) {
        typedef Mesh3 Mesh;
        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;

        MPIcf cfMPI(argc, argv);
        const double cpubegin = CPUtime();

        int nx = 14;

        std::vector<double> uPrint, wPrint, curluPrint, gradwPrint, meanwPrint, h, convuPr, convwPr;
        int iters = 3;

        for (int i = 0; i < iters; ++i) {
            double ox = -1.0+1e-15; // offset to avoid singularity
            double lx = 2 * fabs(ox);
            Mesh Kh(nx, nx, nx, ox, ox, ox, lx, lx, lx);


            Space Lh(Kh, DataFE<Mesh>::P1);
            Fun_h levelSet(Lh, fun_levelSet);
            InterfaceLevelSet<Mesh> interface(Kh, levelSet);
            Normal n;
            Tangent t;

            Space Wh(Kh, DataFE<Mesh>::P1);
            Space Vh(Kh, DataFE<Mesh>::Ned0);
            // Space Uh(Kh, DataFE<Mesh>::RT0);

            ActiveMesh<Mesh> Kh_i(Kh);
            Kh_i.truncate(interface, 1);
            Kh_i.info();
            
            CutSpace WhC(Kh_i, Wh);
            CutSpace VhC(Kh_i, Vh);
            // CutSpace UhC(Kh_i, Uh);

            Lagrange3 DataSpace(2);
            Space Velh(Kh, DataSpace);
            CutSpace VelhC(Kh_i, Velh);

            CutFEM<Mesh> HL(WhC); HL.add(VhC);
            // HL.add(UhC);

            const R h_i  = 1. / (nx - 1);
            const R invh = 1. / h_i;
            MacroElement<Mesh> macro(Kh_i, 0.25);


            Fun_h gradphi(VelhC, fun_force);
            Fun_h u_ex(VelhC, fun_exact_u);
            Fun_h not_exact_form(VhC, fun_closed_form);

            Fun_h w0(WhC, fun_exact_w);
            Fun_h u0(VhC, fun_exact_u);

            FunTest w(WhC, 1, 0), tau(WhC, 1, 0), u(VhC, 3, 0), v(VhC, 3, 0);
            // FunTest p(UhC, 3, 0), q(UhC, 3, 0);

            double wPenParam = 1e0; // 1e-2
            double uPenParam = 1e0; // 1e-2
            R hi1 = pow(h_i, 1);
            R hi3 = pow(h_i, 3);
            R hi5 = pow(h_i, 5);

            double t0 = MPIcf::Wtime();
            // {Assembles LHS and RHS}
            HL.addBilinear(
                innerProduct(w, tau) 
                - innerProduct(u, grad(tau))
            , Kh_i);
            HL.addBilinear(
                innerProduct(grad(w), v)
                + innerProduct(curl(u), curl(v))
            , Kh_i);
            HL.addLinear(
                innerProduct(gradphi.exprList(), v)
            , Kh_i);
            // HL.addBilinear(
            //     // innerProduct(p, q)
            //     - innerProduct(curl(u), q)
            //     + innerProduct(p, curl(v))
            //     // - innerProduct(p, grad(tau))
            //     // + innerProduct(grad(w), q)
            // , Kh_i);

            // {Adds harmonic form Lagrange multiplier with s-stabilization}
            CutFEM<Mesh> lagr(WhC); lagr.add(VhC);
            lagr.addLinear(innerProduct(not_exact_form.exprList(), u), Kh_i);
            lagr.addFaceStabilization(innerProduct(jump(not_exact_form), uPenParam * hi1 * jump(u)), Kh_i, macro);
            lagr.addFaceStabilization(innerProduct(jump(dnormal(not_exact_form)), uPenParam * hi3 * jump(grad(u)*n)), Kh_i, macro);
            Rn lag_row(lagr.rhs_);
            lagr.rhs_ = 0.; 
            lagr.addLinear(innerProduct(not_exact_form.exprList(), v), Kh_i);
            lagr.addFaceStabilization(innerProduct(jump(not_exact_form), uPenParam * hi1 * jump(v)), Kh_i, macro);
            lagr.addFaceStabilization(innerProduct(jump(dnormal(not_exact_form)), uPenParam * hi3 * jump(grad(v)*n)), Kh_i, macro);
            HL.addLagrangeVecToRowAndCol(lag_row, lagr.rhs_, 0);
            // {Adds harmonic form Lagrange multiplier without s-stabilization}
            // HL.addLagrangeMultiplier(
            //     innerProduct(not_exact_form.exprList(), u), 0
            // , Kh_i);
            // {Adds harmonic form Lagrange multiplier with h-stabilization}
            // CutFEM<Mesh> lagr(WhC); lagr.add(VhC);
            // lagr.addLinear(innerProduct(not_exact_form.exprList(), u), Kh_i, INTEGRAL_EXTENSION, 1);
            // Rn lag_row(lagr.rhs_);
            // lagr.rhs_ = 0.; 
            // lagr.addLinear(innerProduct(not_exact_form.exprList(), v), Kh_i, INTEGRAL_EXTENSION, 1);
            // HL.addLagrangeVecToRowAndCol(lag_row, lagr.rhs_, 0);

            // {Adds facet stabilization}
            HL.addFaceStabilization( // [h^(2k+1) h^(2k+1)]
                innerProduct(wPenParam * hi1 * jump(w), jump(tau)) 
                + innerProduct(wPenParam * hi3 * jump(grad(w)*n), jump(grad(tau)*n))

                - innerProduct(uPenParam * hi1 * jump(u), jump(grad(tau)))
                + innerProduct(uPenParam * hi1 * jump(grad(w)), jump(v))

                + innerProduct(uPenParam * hi1 * jump(curl(u)), jump(curl(v)))
                + innerProduct(uPenParam * hi3 * jump(grad(curl(u))*n), jump(grad(curl(v))*n))
                ,
                Kh_i
                , macro
            );

            std::cout << " Time assembly \t" << MPIcf::Wtime() - t0 << std::endl;
            t0 = MPIcf::Wtime();

            // {Export matrix for condition number computation}
            // matlab::Export(HL.mat_[0], "mat" + std::to_string(i) + "Cut.dat");
            // std::cout << " Time export \t" << MPIcf::Wtime() - t0 << std::endl;
            // t0 = MPIcf::Wtime();
            // nx = 2 * nx - 1;
            // continue;

            HL.solve("mumps");
            std::cout << " Time solver \t" << MPIcf::Wtime() - t0 << std::endl;

            // {EXTRACT SOLUTION}
            int nb_dof_WhC  = WhC.get_nb_dof();
            Rn_ data_wh = HL.rhs_(SubArray(WhC.get_nb_dof(), 0));
            Rn_ data_uh = HL.rhs_(SubArray(VhC.get_nb_dof(), nb_dof_WhC));
            Fun_h wh(WhC, data_wh);
            Fun_h uh(VhC, data_uh);

            auto curl_u0 = dy(uh.expr(2)) - dz(uh.expr(1));
            auto curl_u1 = dz(uh.expr(0)) - dx(uh.expr(2));
            auto curl_u2 = dx(uh.expr(1)) - dy(uh.expr(0));

            R normCurlU = L2normCut_2(curl_u0, fun_zero, Kh_i)
                        + L2normCut_2(curl_u1, fun_zero, Kh_i)
                        + L2normCut_2(curl_u2, fun_zero, Kh_i);
            normCurlU = sqrt(normCurlU);

            auto grad_w0 = dx(wh.expr());
            auto grad_w1 = dy(wh.expr());
            auto grad_w2 = dz(wh.expr());

            R wGradDiff = L2normCut_2(grad_w0, fun_force_x, Kh_i)
                        + L2normCut_2(grad_w1, fun_force_y, Kh_i);
                        // + L2normCut_2(grad_w2, 0, Kh_i); // last component is zero
            wGradDiff = sqrt(wGradDiff);

            R meanw = integral(Kh_i, wh, 0);

            R errU      = L2normCut(uh, fun_exact_u, 0, 3);
            R errW      = L2normCut(wh, fun_exact_w, 0, 1);

            // {PLOTTING}
            if (MPIcf::IamMaster()) {
                Fun_h soluh(VhC, fun_exact_u);
                Fun_h solwh(WhC, fun_exact_w);
                // solh.v -= uh.v;
                // solh.v.map(fabs);
                
                Paraview<Mesh> writer(Kh_i, "HLFictitious3D_" + std::to_string(i) + ".vtk");

                writer.add(soluh, "uExact", 0, 3);
                writer.add(solwh, "-divuExact", 0, 1);
                writer.add(uh, "u", 0, 3);
                writer.add(wh, "-divu", 0, 1);
            }

            wPrint.push_back(errW);
            uPrint.push_back(errU);
            curluPrint.push_back(normCurlU);
            gradwPrint.push_back(wGradDiff);
            meanwPrint.push_back(meanw);
            h.push_back(h_i);

            if (i == 0) {
                convwPr.push_back(0);
                convuPr.push_back(0);
            } else {
                convwPr.push_back(log(wPrint[i] / wPrint[i - 1]) / log(h[i] / h[i - 1]));
                convuPr.push_back(log(uPrint[i] / uPrint[i - 1]) / log(h[i] / h[i - 1]));
            }
            nx = 2 * nx - 1;
        }

        std::cout << "\n"
                << std::left << std::setw(10) << std::setfill(' ') << "h" << std::setw(15) << std::setfill(' ')
                << "err w" << std::setw(15) << std::setfill(' ') << "conv w" << std::setw(15) << std::setfill(' ')
                << "err u" << std::setw(15) << std::setfill(' ') << "conv u" << std::setw(15) << std::setfill(' ')
                << "err curlu" << std::setw(15) << std::setfill(' ') << "err gradw" << std::setw(15) << std::setfill(' ')
                << "mean w"
                << "\n"
                << std::endl;
        for (int i = 0; i < uPrint.size(); ++i) {
            std::cout << std::left << std::setprecision(5) << std::setw(10) << std::setfill(' ') << h[i] 
                    << std::setw(15) << std::setfill(' ') << wPrint[i]
                    << std::setw(15) << std::setfill(' ') << convwPr[i]
                    << std::setw(15) << std::setfill(' ') << uPrint[i]
                    << std::setw(15) << std::setfill(' ') << convuPr[i]
                    << std::setw(15) << std::setfill(' ') << curluPrint[i]
                    << std::setw(15) << std::setfill(' ') << gradwPrint[i]
                    << std::setw(15) << std::setfill(' ') << meanwPrint[i]
                    << std::endl;
        }
    }

#endif