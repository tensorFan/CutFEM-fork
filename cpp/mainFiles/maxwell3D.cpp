#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
// #include "../util/cputime.h"
#ifdef USE_MPI
#include "cfmpi.hpp"
#endif

#include "finiteElement.hpp"
#include "baseProblem.hpp"
#include "paraview.hpp"
#include "../num/matlab.hpp"

// #include "../num/gnuplot.hpp"

// f = 1/eps curl j => div f = 0 !
// #define FITTED_WAVE_EIGEN
// #define UNFITTED_WAVE_EIGEN
// #define FITTED_WAVE
// #define UNFITTED_WAVE

// #define FITTED_KIKUCHI_EIGEN
// #define UNFITTED_KIKUCHI_EIGEN
// #define FITTED_KIKUCHI
// #define UNFITTED_KIKUCHI

#define FITTED_3FIELD_EIGEN
// #define UNFITTED_3FIELD_EIGEN
// #define FITTED_3FIELD
// #define UNFITTED_3FIELD


#ifdef FITTED_WAVE_EIGEN

    using namespace globalVariable;
    namespace Data_mueps { // f = 1/eps curl j => div f = 0 !
        R k = 1.;
        R eps = 1.;
        R mu = 1.;

        // Eriks example
        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return pi*pi*sin(pi*z) - sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_u(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_curlu(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return pi*cos(pi*z);
            else
                return 0;
        }

        R fun_0(double *P, int i) {
            return 0;
        }

        R fun_closed_form(double *P, int i) { // = grad 1/r
            R x = P[0], y = P[1], z = P[2];
            R r = sqrt(x*x + y*y + z*z);
            R r3 = r*r*r;
            if (i == 0)
                return x/r3;
            else if (i == 1)
                return y/r3;
            else
                return z/r3;
        }

    }
    using namespace Data_mueps;
    int main(int argc, char **argv) { // (1.10) : curl(1/eps 1/mu curl(u)) - k^2 u = f, f = 1/eps curl j
        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();
        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 7;
        int ny = 7;
        int nz = 7;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 2;
        for (int i = 0; i < iters; ++i) {

            // Mesh3 Kh(nx, ny, nz, 0., 0., 0., M_PI, M_PI, M_PI);
            // Mesh3 Kh("../cpp/mainFiles/meshes/cyli_"+std::to_string(i), MeshFormat::mesh_gmsh);  // sqrt(214)=14.62, sqrt(1268)=35.6089, sqrt(8547)=92.4499
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_"+std::to_string(i), MeshFormat::mesh_gmsh);
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_hole_"+std::to_string(i), MeshFormat::mesh_gmsh);
            Mesh3 Kh("../cpp/mainFiles/meshes/ball_hole_"+std::to_string(i), MeshFormat::mesh_gmsh);
            Kh.info();
            const R hi = 1. / (nx - 1);

            Space Uh(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            R nb_dof = Uh.get_nb_dof();

            Lagrange3 VelocitySpace(2);
            Space Velh(Kh, VelocitySpace);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h curlu0(Velh, fun_exact_curlu);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh);
            CutFEM<Mesh> massRHS(Uh);

            /* Syntax:
            FunTest (fem space, #components, place in space)
            */
            FunTest u(Uh, 3, 0), v(Uh, 3, 0);
            Normal n;

            // (1.10) : curl(1/eps 1/mu curl(u)) - k^2 u = f, f = 1/eps curl j
            R mui = 1./mu;
            R epsi = 1./eps;
            maxwell3D.addBilinear( 
                +innerProduct(epsi * mui * curl(u), curl(v))
                // -innerProduct(k * k * u, v)
            , Kh);
            massRHS.addBilinear( 
                +innerProduct(u, v)
            , Kh);

            // » IF using cube with hole mesh
            Fun_h not_exact_form(Velh, fun_closed_form);
            maxwell3D.addLagrangeMultiplier(
                +innerProduct(not_exact_form.exprList(), v), 0
            , Kh);
            maxwell3D.mat_[0][std::make_pair(nb_dof,nb_dof)] = 0; // For PETsc
            massRHS.addLagrangeMultiplier(
                -innerProduct(not_exact_form.exprList(), 0*v), 0
            , Kh);
            massRHS.mat_[0][std::make_pair(nb_dof,nb_dof)] = 0; // For PETsc

            /* To calculate the number of dofs in the matrix explicitly (ie .mat[0].size())
            int maxRow = -1, maxCol = -1;
            for (auto &entry : maxwell3D.mat_[0]) {
                int i = entry.first.first;
                int j = entry.first.second;
                if (i > maxRow) maxRow = i;
                if (j > maxCol) maxCol = j;
            }
            std::cout << std::max(maxRow, maxCol) + 1 << std::endl; // Assuming zero-based indexing
            */

            // BC
            // Essential n x E (Nitsche)
            // R pp = 1e2;
            // maxwell3D.addBilinear( 
            //     -innerProduct(epsi * mui * curl(u), cross(n, v))
            //     +innerProduct(epsi * mui * cross(n, u), curl(v))
            //     +innerProduct(cross(n, u), 1./hi*pp*cross(n, v))
            // , Kh, INTEGRAL_BOUNDARY);
            // Essential n x E (strong)
            Fun_h fun0(Uh, fun_0);
            maxwell3D.setDirichletHcurl(fun0, Kh);
            massRHS.setDirichletHcurl_RHSMat(fun0, Kh);

            matlab::Export(maxwell3D.mat_[0], "A" + std::to_string(i) + ".dat");
            matlab::Export(massRHS.mat_[0], "B" + std::to_string(i) + ".dat");
            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
            continue;
        }
    }
#endif

#ifdef UNFITTED_WAVE_EIGEN

    using namespace globalVariable;
    namespace Data_CubeLevelset { // f = 1/eps curl j => div f = 0 !
        R k = 1.;
        R eps = 1.;
        R mu = 1.;

        double shift = 0.5*M_PI;
        R sdRoundBox( double* p, double* b, double r ) {
            R qx = std::abs(p[0]) - b[0] + r;
            R qy = std::abs(p[1]) - b[1] + r;
            R qz = std::abs(p[2]) - b[2] + r;

            R val = sqrt(pow(std::max(qx,0.0),2)+pow(std::max(qy,0.0),2)+pow(std::max(qz,0.0),2)) + std::min(std::max(qx,std::max(qy,qz)),0.0) - r;
            // std::cout << val << std::endl;
            return val;
        }
        R fun_levelSet(double *P, int i, int dom) {
            double Pnew[3] = {P[0]-shift, P[1]-shift, P[2]-shift};

            // half-dimensions of box 
            // double b[3] = {0.5*M_PI, 0.5*M_PI, 0.5*M_PI}; // if using cutfem mesh
            double sh_int = 1e-12;
            double b[3] = {0.5*M_PI - sh_int, 0.5*M_PI - sh_int, 0.5*M_PI - sh_int}; // if using GMSH mesh

            // smoothing radius
            double r = 0.025;
            return sdRoundBox(Pnew, b, 0.0);
        }
        R fun_levelSetPLANE(double *P, int i, int dom) {
            return P[2] - M_PI;
        }


        // Eriks example
        R fun_rhs(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return pi*pi*sin(pi*z) - sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_u(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_curlu(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return pi*cos(pi*z);
            else
                return 0;
        }
        R fun_exact_p(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            return x*(x-1)-y*(y-1);
        }

        R fun_closed_form(double *P, int i, int dom) { // = grad 1/r
            R x = P[0], y = P[1], z = P[2];
            R r = sqrt(x*x + y*y + z*z);
            R r3 = r*r*r;
            if (i == 0)
                return x/r3;
            else if (i == 1)
                return y/r3;
            else
                return z/r3;
        }

        R fun_0(double *P, int i, int dom) {
            return 0;
        }

    }

    using namespace Data_CubeLevelset;
    int main(int argc, char **argv) {

        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();

        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 7;
        int ny = 7;
        int nz = 7;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 2;
        for (int i = 0; i < iters; ++i) {
            // R sh = 0.234225;
            Mesh3 Kh(nx, ny, nz, 0, 0, 0, M_PI, M_PI, M_PI+1e-12); // see Mesh3dn.hpp
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_"+std::to_string(i), MeshFormat::mesh_gmsh);
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_hole_"+std::to_string(i), MeshFormat::mesh_gmsh);
            // Kh.info();
            const R hi = 1. / (nx - 1); // 1./(nx-1)

            Lagrange3 VelocitySpace(2);
            Space Uh_(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            Space Vel_h(Kh, VelocitySpace);
            // Space Uh(Khi, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            // Space Velh(Khi, VelocitySpace); Normal n;

            // CutFEM setup
            Space Lh(Kh, DataFE<Mesh>::P1);
            // Fun_h levelSet(Lh, fun_levelSet);
            Fun_h levelSet(Lh, fun_levelSetPLANE);
            InterfaceLevelSet<Mesh> interface(Kh, levelSet);
            Normal n;
            // [Remove exterior]
            ActiveMesh<Mesh> Khi(Kh);
            Khi.truncate(interface, 1);
            Khi.info();
            // Cut spaces
            CutSpace Velh(Khi, Vel_h);
            CutSpace Uh(Khi, Uh_);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h not_exact_form(Velh, fun_closed_form);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh);

            /* Syntax:
            FunTest (fem space, #components, place in space)
            */
            FunTest u(Uh, 3, 0), v(Uh, 3, 0);

            // std::getchar();
            Fun_h fun0(Uh, fun_exact_u);
            Paraview<Mesh> writer(Khi, "maxwell_" + std::to_string(i) + ".vtk");
            writer.add(fun0, "vol", 0, 1);

            R mui = 1./mu;
            R epsi = 1./eps;
            maxwell3D.addBilinear( 
                +innerProduct(epsi * mui * curl(u), curl(v))
                // -innerProduct(k * k * u, v)
            , Khi);
            // BC
            // Essential n x E
            R pp = 1e2;
            maxwell3D.addBilinear( 
                +innerProduct(epsi * mui * curl(u), cross(n, v)) //fitted: -
                -innerProduct(epsi * mui * cross(n, u), curl(v))
                +innerProduct(cross(n, u), 1./hi*pp*cross(n, v))
            , interface);
            maxwell3D.addBilinear( 
                +innerProduct(epsi * mui * curl(u), cross(n, v)) //fitted: -
                -innerProduct(epsi * mui * cross(n, u), curl(v))
                +innerProduct(cross(n, u), 1./hi*pp*cross(n, v))
            , Khi, INTEGRAL_BOUNDARY);
            // [Stabilization]
            double tau_a = 1e-2;
            maxwell3D.addPatchStabilization(
                // A block
                // +innerProduct(tau_a  * jump(u), jump(v)) 
                +innerProduct(tau_a * hi * hi * jump(curl(u)), jump(curl(v)))
            , Khi);

            matlab::Export(maxwell3D.mat_[0], "A" + std::to_string(i) + ".dat");
            // Eigenvalue problem
            CutFEM<Mesh> massRHS(Uh);
            massRHS.addBilinear( 
                +innerProduct(u, v)
            , Khi);
            massRHS.addPatchStabilization(
                // A block
                +innerProduct(tau_a  * jump(u), jump(v)) 
                // +innerProduct(tau_a * jump(curl(u)), jump(curl(v)))
            , Khi);

            matlab::Export(massRHS.mat_[0], "B" + std::to_string(i) + ".dat");
            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
            continue;
        }
        std::cout << "\n"
        << std::left << std::setw(10) << std::setfill(' ') << "h" << std::setw(15) << std::setfill(' ')
        << "err p" << std::setw(15) << std::setfill(' ') << "conv p" << std::setw(15) << std::setfill(' ')
        << "err u" << std::setw(15) << std::setfill(' ') << "conv u" << std::setw(15) << std::setfill(' ')
        << "err divu"
        // << std::setw(15) << std::setfill(' ') << "conv divu"
        // << std::setw(15) << std::setfill(' ') << "err_new divu"
        // << std::setw(15) << std::setfill(' ') << "convLoc divu"
        << std::setw(15) << std::setfill(' ')
        << "err maxdivu"
        // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
        << "\n"
        << std::endl;
    }
#endif

#ifdef FITTED_WAVE

    using namespace globalVariable;
    namespace Data_Old { // f = 1/eps curl j => div f = 0 !
        R k = 1.;
        R eps = 1.;
        R mu = 1.;

        // Monks example
        // R fun_rhs(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return pi*pi*sin(pi*z) - sin(pi*z);
        //     else if (i == 1)
        //         return 0;
        //     else
        //         return 0;
        // }
        // R fun_exact_u(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     R cA = 1./sqrt(7./2);
        //     R ck = 1./(sqrt(13)/2);
        //     if (i == 0)
        //         return cA*cos(ck*(-3./2*x+y));
        //     else if (i == 1)
        //         return cA*3./2*cos(ck*(-3./2*x+y));
        //     else
        //         return cA*1./2*cos(ck*(-3./2*x+y));
        // }

        // Eriks example
        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return pi*pi*sin(pi*z) - sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_u(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_curlu(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return pi*cos(pi*z);
            else
                return 0;
        }

        R fun_0(double *P, int i) {
            return 0;
        }

    }
    namespace Data_Cylinder { // f = 1/eps curl j => div f = 0 !
        R fun_0(double *P, int i) {
            return 0;
        }

        R k = 2.;
        R eps = 1.;
        R mu = 1.;
    
        R cylrad = 0.2;
        R x_0 = 2.405; // first zero of Bessel function J_0(x)
        R q0 = x_0/cylrad; // q0 = h of Cheng
        R ps = 11.8575134409; // propagation speed, sol to 4+x^2 = q0^2
        R fun_exact_u(double *P, int i) { // h^2 = k^2 + \gam^2, E = E_0 exp(-\gam z)
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0)  // -1/q0^2 ps dEz/dx
                return ps/q0 * x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1)
                return ps/q0 * y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return std::cyl_bessel_j(0, q0 * r)*exp(-ps*z);
        }
        R fun_exact_curlu(double *P, int i) { // curlF_theta = -dF/dz, e_theta = -x/r e_x + y/r e_y, J_0' = -J_1 
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0) // dy(Fz) - dz(Fy)
                return (-q0 + ps*ps/q0)*y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1) // dz(Fx) - dx(Fz)
                return (-ps*ps/q0 + q0)*x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return 0; // dx(Fy) - dy(Fx)
        }
        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_divu(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }
        R fun_exact_p(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }

    }
    using namespace Data_Cylinder;
    int main(int argc, char **argv) { // (1.10) : curl(1/eps 1/mu curl(u)) - k^2 u = f, f = 1/eps curl j
        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();
        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 7;
        int ny = 7;
        int nz = 7;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 1;
        for (int i = 0; i < iters; ++i) {

            // Mesh3 Kh(nx, ny, nz, 0., 0., 0., 1., 1., 1.);
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_"+std::to_string(i), MeshFormat::mesh_gmsh);
            //Mesh3 Kh("../cpp/mainFiles/meshes/cube_hole_"+std::to_string(i), MeshFormat::mesh_gmsh);
            Mesh3 Kh("../cpp/mainFiles/meshes/cyli_"+std::to_string(i), MeshFormat::mesh_gmsh);  // sqrt(214)=14.62, sqrt(1268)=35.6089, sqrt(8547)=92.4499
            Kh.info();
            const R hi = 1. / (nx - 1);

            Space Uh(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            // Space Wh_(Kh, DataFE<Mesh>::P1);

            Lagrange3 VelocitySpace(2);
            Space Velh(Kh, VelocitySpace);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h curlu0(Velh, fun_exact_curlu);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh);
            // maxwell3D.add(Wh);

            /* Syntax:
            FunTest (fem space, #components, place in space)
            */
            FunTest u(Uh, 3, 0), v(Uh, 3, 0);
            // FunTest p(Wh, 1, 0), q(Wh, 1, 0);
            Normal n;

            // (1.10) : curl(1/eps 1/mu curl(u)) - k^2 u = f, f = 1/eps curl j
            R mui = 1./mu;
            R epsi = 1./eps;
            maxwell3D.addBilinear( 
                +innerProduct(epsi * mui * curl(u), curl(v))
                -innerProduct(k * k * u, v)
            , Kh);
            maxwell3D.addLinear(
                +innerProduct(fh.exprList(), v)
            , Kh);
            // BC
            // Essential
            R pp = 1e2;
            maxwell3D.addBilinear( 
                -innerProduct(epsi * mui * curl(u), cross(n, v))
                -innerProduct(epsi * mui * cross(n, u), curl(v))
                +innerProduct(cross(n, u), 1./hi*pp*cross(n, v))
            , Kh, INTEGRAL_BOUNDARY);
            maxwell3D.addLinear(
                -innerProduct(cross(n, u0), epsi * mui * curl(v))
                +innerProduct(cross(n, u0), 1./hi*pp*cross(n, v))
            , Kh, INTEGRAL_BOUNDARY);
            // Natural
            // maxwell3D.addLinear(
            //     -innerProduct(cross(n, curlu0), epsi * mui * v)
            // , Kh, INTEGRAL_BOUNDARY);

            matlab::Export(maxwell3D.mat_[0], "mat" + std::to_string(i) + "Cut.dat");
            maxwell3D.solve("umfpack");

            // EXTRACT SOLUTION

            int nb_electric_dof = Uh.get_nb_dof();
            // int nb_lagrange_dof = Wh.get_nb_dof();

            Rn_ data_uh = maxwell3D.rhs_(SubArray(nb_electric_dof, 0));
            // Rn_ data_ph = maxwell3D.rhs_(SubArray(nb_lagrange_dof, nb_electric_dof)); // Rn_ data_uh = stokes.rhs_(SubArray(nb_vort_dof+nb_flux_dof,nb_vort_dof));

            Fun_h uh(Uh, data_uh);
            // Fun_h ph(Wh, data_ph);

            auto uh_0dx = dx(uh.expr(0));
            auto uh_1dy = dy(uh.expr(1));
            auto uh_2dz = dz(uh.expr(2));

            // [Paraview]
            {
                Fun_h solu(Velh, fun_exact_u);

                Fun_h soluErr(Uh, fun_exact_u);
                soluErr.v -= uh.v;
                soluErr.v.map(fabs);

                // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

                Paraview<Mesh> writer(Kh, "maxwell_" + std::to_string(i) + ".vtk");
                writer.add(uh, "electric_field", 0, 3);
                // writer.add(uh_0dx + uh_1dy + uh_2dz, "divergence");
                writer.add(uh_2dz, "divergence");
                writer.add(solu, "electric_field_exact", 0, 3);
                writer.add(soluErr, "electric_field_error", 0, 3);
            }

            R errU      = L2norm(uh, fun_exact_u, 0, 3);
            R errDiv    = L2norm(uh_0dx + uh_1dy + uh_2dz, fun_0, Kh);
            Space P1_(Kh, DataFE<Mesh>::P1);
            R maxErrDiv = maxNormEdges(uh_0dx + uh_1dy + uh_2dz, Kh, P1_);
            // R maxErrDiv = maxNorm(uh_0dx, Kh);

            // Face jump errors as measure of weak divergence error
            Space P0_(Kh, DataFE<Mesh>::P0); FunTest p(P0_, 1, 0), q(P0_, 1, 0);
            CutFEM<Mesh> err_face_jumps(P0_);
            err_face_jumps.addLinear(
                +innerProduct(uh.exprList(), uh.exprList() * jump(q))
            , Kh, INTEGRAL_INNER_FACET);
            errDiv = sqrt(abs(err_face_jumps.rhs_.sum()));

            h.push_back(hi);
            ul2.push_back(errU);
            divl2.push_back(errDiv);
            divmax.push_back(maxErrDiv);
            if (i == 0) {
                convu.push_back(0);
            } else {
                convu.push_back(log(ul2[i] / ul2[i - 1]) / log(h[i] / h[i - 1]));
            }
            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
        }
        std::cout << "\n"
        << std::left << std::setw(10) << std::setfill(' ') << "h" << std::setw(15) << std::setfill(' ')
        << "err u" << std::setw(15) << std::setfill(' ') << "conv u" << std::setw(15) << std::setfill(' ')
        << "err divu"
        // << std::setw(15) << std::setfill(' ') << "conv divu"
        << std::setw(15) << std::setfill(' ')
        << "err maxdivu"
        // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
        << "\n"
        << std::endl;
        for (int i = 0; i < h.size(); ++i) {
            std::cout << std::left << std::setw(10) << std::setfill(' ') << h[i] << std::setw(15) << std::setfill(' ')
            << ul2[i] << std::setw(15) << std::setfill(' ') << convu[i] << std::setw(15) << std::setfill(' ')
            << divl2[i]
            // << std::setw(15) << std::setfill(' ') << convdivPr[i]
            << std::setw(15) << std::setfill(' ')
            << divmax[i]
            // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
            << std::endl;
        }
    }
#endif

#ifdef UNFITTED_WAVE

    using namespace globalVariable;
    namespace Data_Old {
        R k = 1.;
        R eps_r = 1.;
        R mu = 1.;

        R3 shift(0.5, 0.5, 0.5);

        R fun_levelSet(double *P, int i) {
            return (P[0] - shift.x) * (P[0] - shift.x) + (P[1] - shift.y) * (P[1] - shift.y) +
                (P[2] - shift.z) * (P[2] - shift.z) - 0.35 * 0.35 + Epsilon;
        }

        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 2 * pi * pi * sin(y * pi) * sin(z * pi) -
                    eps_r * sin(y * pi) * sin(z * pi) * k * k; //+ 2 * (P[0] - shift.x) - eps_r * (2 * x - 1);
            else if (i == 1)
                return 2 * pi * pi * sin(x * pi) * sin(z * pi) -
                    eps_r * sin(x * pi) * sin(z * pi) * k * k; //+ 2 * (P[1] - shift.y) - eps_r * (2 * y - 1);
            else
                return 2 * pi * pi * sin(x * pi) * sin(y * pi) -
                    eps_r * sin(x * pi) * sin(y * pi) * k * k; //+ 2 * (P[2] - shift.z) - eps_r * (2 * z - 1);
        }
        R fun_exact_u(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi * y) * sin(pi * z);
            else if (i == 1)
                return sin(pi * x) * sin(pi * z);
            else
                return sin(pi * x) * sin(pi * y);
        }
        R fun_exact_curlu(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return pi*(cos(pi*y) - cos(pi*z))*sin(pi*x);
            else if (i == 1)
                return pi*(-cos(pi*x) + cos(pi*z))*sin(pi*y);
            else
                return pi*(cos(pi*x) - cos(pi*y))*sin(pi*z);
        }
        R fun_exact_p(double *P, int i, int dom) {
            return (P[0] - shift.x) * (P[0] - shift.x) + (P[1] - shift.y) * (P[1] - shift.y) +
                (P[2] - shift.z) * (P[2] - shift.z) - 0.35 * 0.35;
        }

        // Eriks example
        // R fun_rhs(double *P, int i, int dom) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return pi*pi*sin(pi*z) - sin(pi*z);
        //     else if (i == 1)
        //         return 0;
        //     else
        //         return 0;
        // }
        // R fun_exact_u(double *P, int i, int dom) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return sin(pi*z);
        //     else if (i == 1)
        //         return 0;
        //     else
        //         return 0;
        // }
        // R fun_exact_curlu(double *P, int i, int dom) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return 0;
        //     else if (i == 1)
        //         return pi*cos(pi*z);
        //     else
        //         return 0;
        // }

    }
    namespace Data_Cylinder {
        R fun_0(double *P, int i) {
            return 0;
        }

        R k = 2.;
        R eps = 1.;
        R mu = 1.;

        R cylrad = 0.2;
        R cylheight = 0.35; // half of cylinder height = 0.35
        R3 shift(0.0, 0.0, cylheight);
        R sdCylinder(double *p, float h, float r) {
            R px = p[0], py = p[1], pz = p[2];
            float length_p_xz = sqrt(px * px + pz * pz);
            float abs_p_y = fabs(py);
            float dx = fabs(length_p_xz) - r;
            float dy = fabs(abs_p_y) - h;
            
            float maxD = std::max(dx, dy);
            float minMaxD = std::min(maxD, 0.0f);
            
            float maxDx = std::max(dx, 0.0f);
            float maxDy = std::max(dy, 0.0f);
            float lengthMaxD = sqrt(maxDx * maxDx + maxDy * maxDy);
            
            return minMaxD + lengthMaxD;
        }
        R fun_levelSet(double *P, int i) {
            R3 pcyl(P[0]-shift.x,P[2]-shift.z,P[1]-shift.y); // cyl coords: (x,z,y)
            return sdCylinder(pcyl, cylheight, cylrad);
        }

        R x_0 = 2.405; // first zero of Bessel function J_0(x)
        R q0 = x_0/cylrad; // q0 = h of Cheng
        R ps = 11.8575134409; // propagation speed, sol to 4+x^2 = q0^2
        R fun_exact_u(double *P, int i, int dom) { // h^2 = k^2 + \gam^2, E = E_0 exp(-\gam z)
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0)  // -1/q0^2 ps dEz/dx
                return ps/q0 * x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1)
                return ps/q0 * y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return std::cyl_bessel_j(0, q0 * r)*exp(-ps*z);
        }
        R fun_exact_curlu(double *P, int i, int dom) { // curlF_theta = -dF/dz, e_theta = -x/r e_x + y/r e_y, J_0' = -J_1 
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0) // dy(Fz) - dz(Fy)
                return (-q0 + ps*ps/q0)*y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1) // dz(Fx) - dx(Fz)
                return (-ps*ps/q0 + q0)*x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return 0; // dx(Fy) - dy(Fx)
        }
        R fun_rhs(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_divu(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }
        R fun_exact_p(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }

    }
    using namespace Data_Cylinder;

    int main(int argc, char **argv) {
        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();
        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 7;
        int ny = 7;
        int nz = 7;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 3;
        for (int i = 0; i < iters; ++i) {
            R sh = M_PI*3e-2;
            Mesh3 Kh(nx, ny, nz, -0.2-sh, -0.2-sh, 0.0-sh, 0.4+sh, 0.4+sh, 2.01*cylheight+sh);
            const R hi = 1. / (nx - 1);

            Space Uh_(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1

            Lagrange3 VelocitySpace(2);
            Space Vel_h(Kh, VelocitySpace);

            Space Lh(Kh, DataFE<Mesh>::P1);
            Fun_h levelSet(Lh, fun_levelSet);
            InterfaceLevelSet<Mesh> interface(Kh, levelSet);
            Normal n;

            // [Remove exterior]
            ActiveMesh<Mesh> Khi(Kh);
            Khi.truncate(interface, 1);
            Khi.info();

            CutSpace Velh(Khi, Vel_h);
            CutSpace Uh(Khi, Uh_);
            // CutSpace Wh(Khi, Wh_);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h curlu0(Velh, fun_exact_curlu);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh);

            /* Syntax:
            FunTest (fem space, #components, place in space)
            */
            FunTest u(Uh, 3, 0), v(Uh, 3, 0);

            maxwell3D.addBilinear( 
                +innerProduct(1./mu * 1./eps * curl(u), curl(v))
                -innerProduct(k * k * u, v)
            , Khi);
            maxwell3D.addLinear(
                +innerProduct(fh.exprList(), v)
            , Khi);
            R pp = 1e2;
            // maxwell3D.addBilinear( 
            //     +innerProduct(1./mu * curl(u), cross(n, v))
            //     -innerProduct(1./mu * cross(n, u), curl(v))
            //     +innerProduct(cross(n, u), 1./hi*pp*cross(n, v))
            // , interface);
            // maxwell3D.addLinear(
            //     -innerProduct(cross(n, u0), 1./mu * curl(v))
            //     +innerProduct(cross(n, u0), 1./hi*pp*cross(n, v))
            // , interface);
            maxwell3D.addLinear( // so weird: unfitted changes the sign
                +innerProduct(cross(n, curlu0), 1./mu * 1./eps * v)
            , interface);

            // [Stabilization]
            double tau_a = 1e0;
            maxwell3D.addPatchStabilization(
            // W block
            - innerProduct(tau_a * pow(hi, 0) * jump(u), jump(v)) 
            , Khi);
            // maxwell3D.addFaceStabilization(
            //     + innerProduct(tau_a * pow(hi, 1) * jump(u), jump(v)) 
            //     + innerProduct(tau_a * pow(hi, 3) * jump(grad(u)*n), jump(grad(v)*n)) 
            // , Khi);

            matlab::Export(maxwell3D.mat_[0], "mat" + std::to_string(i) + "Cut.dat");
            maxwell3D.solve("umfpack");

            // EXTRACT SOLUTION

            int nb_electric_dof = Uh.get_nb_dof();
            Rn_ data_uh = maxwell3D.rhs_(SubArray(nb_electric_dof, 0));
            Fun_h uh(Uh, data_uh);

            auto uh_0dx = dx(uh.expr(0));
            auto uh_1dy = dy(uh.expr(1));
            auto uh_2dz = dz(uh.expr(2));

            // [Paraview]
            {
                Fun_h solu(Velh, fun_exact_u);
                Fun_h soluErr(Uh, fun_exact_u);

                soluErr.v -= uh.v;
                soluErr.v.map(fabs);

                // Fun_h divSolh(Wh, fun_div);
                // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

                Paraview<Mesh> writer(Khi, "maxwell_" + std::to_string(i) + ".vtk");
                writer.add(uh, "electric_field", 0, 3);
                // writer.add(ph, "pressure", 0, 1);
                // writer.add(dx_uh0+dy_uh1, "divergence");
                // writer.add(femSol_0dx+femSol_1dy+fflambdah, "divergence");
                // writer.add(solp, "pressureExact", 0, 1);
                writer.add(solu, "electric_field_exact", 0, 3);
                writer.add(soluErr, "electric_field_error", 0, 3);
            }
            R errU           = L2normCut(uh, fun_exact_u, 0, 3);
            double errDiv    = L2normCut(uh_0dx + uh_1dy + uh_2dz, Khi);
            double maxErrDiv = maxNormCut(uh_0dx + uh_1dy + uh_2dz, Khi);

            ul2.push_back(errU);
            divl2.push_back(errDiv);
            divmax.push_back(maxErrDiv);
            h.push_back(hi);
            if (i == 0) {
                convu.push_back(0);
            } else {
                convu.push_back(log(ul2[i] / ul2[i - 1]) / log(h[i] / h[i - 1]));
            }
            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
        }
        std::cout << "\n"
        << std::left << std::setw(10) << std::setfill(' ') << "h" << std::setw(15) << std::setfill(' ')
        << "err u" << std::setw(15) << std::setfill(' ') << "conv u" << std::setw(15) << std::setfill(' ')
        << "err divu"
        // << std::setw(15) << std::setfill(' ') << "conv divu"
        << std::setw(15) << std::setfill(' ')
        << "err maxdivu"
        // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
        << "\n"
        << std::endl;
        for (int i = 0; i < h.size(); ++i) {
            std::cout << std::left << std::setw(10) << std::setfill(' ') << h[i] << std::setw(15) << std::setfill(' ')
            << ul2[i] << std::setw(15) << std::setfill(' ') << convu[i] << std::setw(15) << std::setfill(' ')
            << divl2[i]
            // << std::setw(15) << std::setfill(' ') << convdivPr[i]
            << std::setw(15) << std::setfill(' ')
            << divmax[i]
            // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
            << std::endl;
        }
    }

#endif


#ifdef FITTED_KIKUCHI_EIGEN

    using namespace globalVariable;
    namespace Data_mueps {
        R k = 1.;
        R eps = 1.;
        R mu = 1.;

        // Monks example
        // R fun_rhs(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return pi*pi*sin(pi*z) - sin(pi*z);
        //     else if (i == 1)
        //         return 0;
        //     else
        //         return 0;
        // }
        // R fun_exact_u(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     R cA = 1./sqrt(7./2);
        //     R ck = 1./(sqrt(13)/2);
        //     if (i == 0)
        //         return cA*cos(ck*(-3./2*x+y));
        //     else if (i == 1)
        //         return cA*3./2*cos(ck*(-3./2*x+y));
        //     else
        //         return cA*1./2*cos(ck*(-3./2*x+y));
        // }

        // Eriks example
        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return pi*pi*sin(pi*z) - sin(pi*z) - (2*x-1);
            else if (i == 1)
                return 2*y-1;
            else
                return 0;
        }
        R fun_exact_u(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_curlu(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return pi*cos(pi*z);
            else
                return 0;
        }
        R fun_exact_p(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return x*(x-1)-y*(y-1);
        }
        R fun_0(double *P, int i) {
            return 0;
        }
        R fun_closed_form(double *P, int i) { // = grad 1/r
            R x = P[0], y = P[1], z = P[2];
            R r = sqrt(x*x + y*y + z*z);
            R r3 = r*r*r;
            if (i == 0)
                return x/r3;
            else if (i == 1)
                return y/r3;
            else
                return z/r3;
        }
    }
    using namespace Data_mueps;
    int main(int argc, char **argv) {
        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();
        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 7;
        int ny = 7;
        int nz = 7;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 2;
        for (int i = 0; i < iters; ++i) {
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_"+std::to_string(i), MeshFormat::mesh_gmsh);  
            Mesh3 Kh("../cpp/mainFiles/meshes/cube_hole_"+std::to_string(i), MeshFormat::mesh_gmsh);  
            // Mesh3 Kh(nx, ny, nz, 0., 0., 0., M_PI, M_PI, M_PI);
            Kh.info();
            const R hi = 1. / (nx - 1);

            Space Uh(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            Space Wh(Kh, DataFE<Mesh>::P1);
            R nb_dof = Uh.get_nb_dof() + Wh.get_nb_dof();

            Lagrange3 VelocitySpace(2);
            Space Velh(Kh, VelocitySpace);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h curlu0(Velh, fun_exact_curlu);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh);
            maxwell3D.add(Wh);

            /* Syntax:
            FunTest (fem space, #components, place in space)
            */
            FunTest u(Uh, 3, 0), v(Uh, 3, 0);
            FunTest p(Wh, 1, 0), q(Wh, 1, 0);
            Normal n;

            maxwell3D.addBilinear( 
                +innerProduct(1./mu * 1./eps * curl(u), curl(v))
                +innerProduct(0*p,q) // for PETSc

                -innerProduct(grad(p), v)
                -innerProduct(u, grad(q))
            , Kh);

            // RHS MAT
            CutFEM<Mesh> massRHS(Uh); massRHS.add(Wh);
            R el_area = hi*hi*hi;
            R regularizer = 1e-12/el_area; // default= 1e-12/el_area
            massRHS.addBilinear( 
                +innerProduct(u, v)
                +innerProduct(regularizer*p, q)
            , Kh);

            // IF using cube with hole mesh
            // Fun_h not_exact_form(Velh, fun_closed_form);
            // maxwell3D.addLagrangeMultiplier(
            //     +innerProduct(not_exact_form.exprList(), v), 0
            // , Kh);
            // maxwell3D.mat_[0][std::make_pair(nb_dof,nb_dof)] = 0; // For PETsc
            // // FEM<Mesh> lagr(Uh); lagr.add(Vh); lagr.add(Wh);
            // // lagr.addLinear(innerProduct(not_exact_form.exprList(), u), Kh);
            // // Rn lag_row(lagr.rhs_); 
            // // lagr.rhs_ = 0.; 
            // // lagr.addLinear(innerProduct(not_exact_form.exprList(), v), Kh);
            // // maxwell3D.addLagrangeVecToRowAndCol(lag_row, lagr.rhs_, 0);
            // massRHS.addLagrangeMultiplier(
            //     -innerProduct(not_exact_form.exprList(), 0*v), 0
            // , Kh);
            // massRHS.mat_[0][std::make_pair(nb_dof,nb_dof)] = 0; // For PETsc

            // » BC
            // »» Essential strong
            // maxwell3D.setDirichletHoneAndHcurl(Wh, Uh, Kh);
            // massRHS.setDirichletHoneAndHcurl_RHSMat(Wh, Uh, Kh);

            matlab::Export(maxwell3D.mat_[0], "A" + std::to_string(i) + ".dat");
            matlab::Export(massRHS.mat_[0], "B" + std::to_string(i) + ".dat");
            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
            continue;
        }
    }
#endif

#ifdef UNFITTED_KIKUCHI_EIGEN

    using namespace globalVariable;
    namespace Data_CubeLevelset { // f = 1/eps curl j => div f = 0 !
        R k = 1.;
        R eps = 1.;
        R mu = 1.;

        double shift = 0.5*M_PI;
        R sdRoundBox( double* p, double* b, double r ) {
            R qx = std::abs(p[0]) - b[0] + r;
            R qy = std::abs(p[1]) - b[1] + r;
            R qz = std::abs(p[2]) - b[2] + r;

            R val = sqrt(pow(std::max(qx,0.0),2)+pow(std::max(qy,0.0),2)+pow(std::max(qz,0.0),2)) + std::min(std::max(qx,std::max(qy,qz)),0.0) - r;
            // std::cout << val << std::endl;
            return val;
        }
        R fun_levelSet(double *P, int i, int dom) {
            double Pnew[3] = {P[0]-shift, P[1]-shift, P[2]-shift};

            // half-dimensions of box 
            // double b[3] = {0.5*M_PI, 0.5*M_PI, 0.5*M_PI}; // if using cutfem mesh
            double sh_int = 1e-12;
            double b[3] = {0.5*M_PI - sh_int, 0.5*M_PI - sh_int, 0.5*M_PI - sh_int}; // if using GMSH mesh

            // smoothing radius
            double r = 0.025;
            return sdRoundBox(Pnew, b, 0.0);
        }
        R fun_levelSetPLANE(double *P, int i, int dom) {
            return P[2] - M_PI;
        }


        // Eriks example
        R fun_rhs(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return pi*pi*sin(pi*z) - sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_u(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_curlu(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return pi*cos(pi*z);
            else
                return 0;
        }
        R fun_exact_p(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            return x*(x-1)-y*(y-1);
        }

        R fun_closed_form(double *P, int i, int dom) { // = grad 1/r
            R x = P[0], y = P[1], z = P[2];
            R r = sqrt(x*x + y*y + z*z);
            R r3 = r*r*r;
            if (i == 0)
                return x/r3;
            else if (i == 1)
                return y/r3;
            else
                return z/r3;
        }

        R fun_0(double *P, int i, int dom) {
            return 0;
        }

    }

    using namespace Data_CubeLevelset;
    int main(int argc, char **argv) {

        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();

        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 7;
        int ny = 7;
        int nz = 7;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 2;
        for (int i = 0; i < iters; ++i) {
            // R sh = 0.234225;
            Mesh3 Kh(nx, ny, nz, 0, 0, 0, M_PI, M_PI, M_PI+1e-12); // see Mesh3dn.hpp
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_"+std::to_string(i), MeshFormat::mesh_gmsh);
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_hole_"+std::to_string(i), MeshFormat::mesh_gmsh);
            // Kh.info();
            const R hi = 1. / (nx - 1); // 1./(nx-1)

            Lagrange3 VelocitySpace(2);
            Space Uh_(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            Space Wh_(Kh, DataFE<Mesh>::P1);
            Space Vel_h(Kh, VelocitySpace);
            // Space Uh(Khi, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            // Space Velh(Khi, VelocitySpace); Normal n;

            // CutFEM setup
            Space Lh(Kh, DataFE<Mesh>::P1);
            // Fun_h levelSet(Lh, fun_levelSet);
            Fun_h levelSet(Lh, fun_levelSetPLANE);
            InterfaceLevelSet<Mesh> interface(Kh, levelSet);
            Normal n;
            // [Remove exterior]
            ActiveMesh<Mesh> Khi(Kh);
            Khi.truncate(interface, 1);
            Khi.info();
            // Cut spaces
            CutSpace Velh(Khi, Vel_h);
            CutSpace Uh(Khi, Uh_);
            CutSpace Wh(Khi, Wh_);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h not_exact_form(Velh, fun_closed_form);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh); maxwell3D.add(Wh);

            /* Syntax:
            FunTest (fem space, #components, place in space)
            */
            FunTest u(Uh, 3, 0), v(Uh, 3, 0);
            FunTest p(Wh, 1, 0), q(Wh, 1, 0);

            // std::getchar();
            Fun_h fun0(Uh, fun_exact_u);
            Paraview<Mesh> writer(Khi, "maxwell_" + std::to_string(i) + ".vtk");
            writer.add(fun0, "vol", 0, 1);

            R mui = 1./mu;
            R epsi = 1./eps;
            maxwell3D.addBilinear( 
                +innerProduct(epsi * mui * curl(u), curl(v))
                // -innerProduct(k * k * u, v)

                +innerProduct(grad(p), v)
                +innerProduct(u, grad(q))
            , Khi);
            // BC
            // Essential n x E
            R pp = 1e2;
            maxwell3D.addBilinear( 
                +innerProduct(epsi * mui * curl(u), cross(n, v)) //fitted: -
                -innerProduct(epsi * mui * cross(n, u), curl(v))
                +innerProduct(cross(n, u), 1./hi*pp*cross(n, v))

                +innerProduct(p, 1./hi*pp*q)
            , interface);
            maxwell3D.addBilinear( // if using plane levelset
                +innerProduct(epsi * mui * curl(u), cross(n, v)) //fitted: -
                -innerProduct(epsi * mui * cross(n, u), curl(v))
                +innerProduct(cross(n, u), 1./hi*pp*cross(n, v))

                +innerProduct(p, 1./hi*pp*q)
            , Khi, INTEGRAL_BOUNDARY);
            // [Stabilization]
            double tau_a = 1e-1; //1e-2
            double tau_p = 1e-1; //1e-2
            maxwell3D.addPatchStabilization(
                // A block
                // +innerProduct(tau_a  * jump(u), jump(v)) 
                +innerProduct(tau_a * hi * hi * jump(curl(u)), jump(curl(v)))

                +innerProduct(tau_p * jump(p), jump(q))
            , Khi);

            matlab::Export(maxwell3D.mat_[0], "A" + std::to_string(i) + ".dat");
            // Eigenvalue problem
            CutFEM<Mesh> massRHS(Uh); massRHS.add(Wh);
            R el_area = hi*hi*hi;
            R regularizer = 1e-12/el_area;
            massRHS.addBilinear( 
                +innerProduct(u, v)
                +innerProduct(regularizer*p, q)
            , Khi);
            massRHS.addPatchStabilization(
                // A block
                +innerProduct(tau_a * jump(u), jump(v)) 
                // +innerProduct(tau_a * jump(curl(u)), jump(curl(v)))
            , Khi);

            matlab::Export(massRHS.mat_[0], "B" + std::to_string(i) + ".dat");
            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
            continue;
        }
    }
#endif

#ifdef FITTED_KIKUCHI

    using namespace globalVariable;
    namespace Data_Old {
        R k = 1.;
        R eps = 1.;
        R mu = 1.;

        // Monks example
        // R fun_rhs(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return pi*pi*sin(pi*z) - sin(pi*z);
        //     else if (i == 1)
        //         return 0;
        //     else
        //         return 0;
        // }
        // R fun_exact_u(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     R cA = 1./sqrt(7./2);
        //     R ck = 1./(sqrt(13)/2);
        //     if (i == 0)
        //         return cA*cos(ck*(-3./2*x+y));
        //     else if (i == 1)
        //         return cA*3./2*cos(ck*(-3./2*x+y));
        //     else
        //         return cA*1./2*cos(ck*(-3./2*x+y));
        // }

        // Eriks example
        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return pi*pi*sin(pi*z) - sin(pi*z) - (2*x-1);
            else if (i == 1)
                return 2*y-1;
            else
                return 0;
        }
        R fun_exact_u(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_curlu(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return pi*cos(pi*z);
            else
                return 0;
        }
        R fun_exact_p(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return x*(x-1)-y*(y-1);
        }

        R fun_0(double *P, int i) {
            return 0;
        }

    }
    namespace Data_Cylinder { // f = 1/eps curl j => div f = 0 !
        R fun_0(double *P, int i) {
            return 0;
        }

        R k = 2.;
        R eps = 1.;
        R mu = 1.;
    
        R cylrad = 0.2;
        R x_0 = 2.405; // first zero of Bessel function J_0(x)
        R q0 = x_0/cylrad; // q0 = h of Cheng
        R ps = 11.8575134409; // propagation speed, sol to 4+x^2 = q0^2
        R fun_exact_u(double *P, int i) { // h^2 = k^2 + \gam^2, E = E_0 exp(-\gam z)
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0)  // -1/q0^2 ps dEz/dx
                return ps/q0 * x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1)
                return ps/q0 * y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return std::cyl_bessel_j(0, q0 * r)*exp(-ps*z);
        }
        R fun_exact_curlu(double *P, int i) { // curlF_theta = -dF/dz, e_theta = -x/r e_x + y/r e_y, J_0' = -J_1 
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0) // dy(Fz) - dz(Fy)
                return (-q0 + ps*ps/q0)*y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1) // dz(Fx) - dx(Fz)
                return (-ps*ps/q0 + q0)*x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return 0; // dx(Fy) - dy(Fx)
        }
        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_divu(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }
        R fun_exact_p(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }

    }
    namespace Data_Cylinder_Sebastian {

        // double k = 2.;
        double k = std::sqrt(1.6);
        double eps = 1.;
        double mu = 1.;

        double cyl_radius = 0.2;
        double cyl_height = 0.35;

        double shift[3] = {0.0, 0.0, cyl_height};

        double sd_cylinder(double *P, float h, float r) {
                double px = P[0], py = P[1], pz = P[2];
                float length_p_xz = std::sqrt(px * px + pz * pz);
                float abs_p_y     = std::fabs(py);
                float dx          = std::fabs(length_p_xz) - r;
                float dy          = std::fabs(abs_p_y) - h;
                
                float maxD         = std::max(dx, dy);
                float minMaxD      = std::min(maxD, 0.0f);
                
                float maxDx        = std::max(dx, 0.0f);
                float maxDy        = std::max(dy, 0.0f);
                float lengthMaxD   = std::sqrt(maxDx * maxDx + maxDy * maxDy);
                
                return minMaxD + lengthMaxD;
        }

        double phi(double *P, int i) {
            double pcyl[3] = {P[0]-shift[0], P[2]-shift[2], P[1]-shift[1]};
            return sd_cylinder(pcyl, cyl_height, cyl_radius);
        }

        double x0 = 2.405;                      // first zero of Bessel function J_0(x)
        double q0 = x0/cyl_radius;              // q0 = h of Cheng
        double ps = std::sqrt(q0*q0 - k*k);     // propagation speed, sol to k^2+x^2 = q0^2
        
        // fun_exact_u = electric_field
        double fun_exact_u(double *P, int i) { // h^2 = k^2 + \gam^2, E = E_0 exp(-\gam z)
            double x = P[0], y = P[1], z = P[2];
            double r = std::sqrt(x*x + y*y);
            if (i == 0)  
                return ps/q0 * x/r * std::cyl_bessel_j(1, q0*r)*exp(-ps*z);
            else if (i == 1)
                return ps/q0 * y/r * std::cyl_bessel_j(1, q0*r)*exp(-ps*z);
            else
                return std::cyl_bessel_j(0, q0*r)*exp(-ps*z) + r*r/4;
        }

        // fun_exact_curlu = curl_e
        double fun_exact_curlu(double *P, int i) {
            double x = P[0], y = P[1], z = P[2];
            double r = std::sqrt(x*x + y*y);
            if (i == 0)  
                return y/2 + (y*std::exp(-ps*z)*(ps*ps - q0*q0)*std::cyl_bessel_j(1, q0*r))/(q0*r);
            else if (i == 1)
                return - x/2 - (x*std::exp(-ps*z)*(ps*ps - q0*q0)*std::cyl_bessel_j(1, q0*r))/(q0*r);
            else
                return 0.;
        }

        double magnetic_induction(double *P, int i) {
            // This is actually i*b

            double x = P[0], y = P[1], z = P[2];
            double r = std::sqrt(x*x + y*y);

            if (i == 0)  
                return -(y/2 + (y*std::exp(-ps*z)*(ps*ps - q0*q0)*std::cyl_bessel_j(1, q0*r))/(q0*r))/k;
            else if (i == 1)
                return (x/2 + (x*std::exp(-ps*z)*(ps*ps - q0*q0)*std::cyl_bessel_j(1, q0*r))/(q0*r))/k;
            else    
                return 0.;
        }

        double fun_rhs(double *P, int i) {
            // This is actually -k/eps * ij

            double x = P[0], y = P[1], z = P[2];
            double r = std::sqrt(x*x + y*y);

            if (i == 0)
                return 0.;
            else if (i == 1)
                return 0.;
            else
                return -(1+ r*r);
        }

        double lagrange_multiplier(double *P, int i) {
            return 0.;
        }

        double div_e(double *P, int i, int dom) {return 0.;}

        double fun_0(double *P, int i) {return 0.;}

        R fun_exact_p(double *P, int i) {return 0;}
    }
    namespace Data_Easy_Cube {
        R fun_0(double *P, int i) {
            return 0;
        }

        R k = 1.;
        R eps = 1.;
        R mu = 1.;

        R fun_exact_u(double *P, int i) { 
            R x = P[0], y = P[1], z = P[2];
            if (i == 0) 
                return x;
            else if (i == 1)
                return -2*y;
            else
                return z;
        }
        R fun_exact_curlu(double *P, int i) { 
            R x = P[0], y = P[1], z = P[2];
            if (i == 0) // dy(Fz) - dz(Fy)
                return 0;
            else if (i == 1) // dz(Fx) - dx(Fz)
                return 0;
            else
                return 0; // dx(Fy) - dy(Fx)
        }
        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return -k*k*x;
            else if (i == 1)
                return +2*k*k*y;
            else
                return -k*k*z;
        }
        R fun_exact_divu(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }
        R fun_exact_p(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }

    }
    // using namespace Data_Cylinder_Sebastian;
    // using namespace Data_Cylinder;
    using namespace Data_Easy_Cube;
    int main(int argc, char **argv) {
        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();
        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 7;
        int ny = 7;
        int nz = 7;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 2;
        for (int i = 0; i < iters; ++i) {
            // Mesh3 Kh("../cpp/mainFiles/meshes/cyli_"+std::to_string(i), MeshFormat::mesh_gmsh);  // sqrt(214)=14.62, sqrt(1268)=35.6089, sqrt(8547)=92.4499
            Mesh3 Kh(nx, ny, nz, 0., 0., 0., 1., 1., 1.);
            Kh.info();
            const R hi = 1. / (nx - 1);

            Space Uh(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            Space Wh(Kh, DataFE<Mesh>::P1);

            Lagrange3 VelocitySpace(2);
            Space Velh(Kh, VelocitySpace);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h curlu0(Velh, fun_exact_curlu);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh);
            maxwell3D.add(Wh);

            /* Syntax:
            FunTest (fem space, #components, place in space)
            */
            FunTest u(Uh, 3, 0), v(Uh, 3, 0);
            FunTest p(Wh, 1, 0), q(Wh, 1, 0);
            Normal n;

            maxwell3D.addBilinear( 
                +innerProduct(1./mu * 1./eps * curl(u), curl(v))
                -innerProduct(k * k * u, v)

                -innerProduct(grad(p), v)
                -innerProduct(u, grad(q))
            , Kh);
            maxwell3D.addLinear(
                +innerProduct(fh.exprList(), v)
            , Kh);
            // BC
            // Essential weak (u)
            // R pp = 1e2;
            // maxwell3D.addBilinear( 
            //     -innerProduct(1./mu * curl(u), cross(n, v))
            //     -innerProduct(1./mu * cross(n, u), curl(v))
            //     +innerProduct(cross(n, u), 1./hi*pp*cross(n, v))
            //     // +innerProduct(u, 1./hi*pp*v)
            // , Kh, INTEGRAL_BOUNDARY);
            // maxwell3D.addLinear(
            //     -innerProduct(cross(n, u0), 1./mu * curl(v))
            //     +innerProduct(cross(n, u0), 1./hi*pp*cross(n, v))
            //     // +innerProduct(u0.exprList(), 1./hi*pp*v)
            // , Kh, INTEGRAL_BOUNDARY);
            // Natural (u)
            maxwell3D.addLinear(
                -innerProduct(cross(n, curlu0), 1./mu * 1./eps * v)
            , Kh, INTEGRAL_BOUNDARY);

            // Weak (p)
            // R pp = 1e2;
            // maxwell3D.addBilinear( // ensuring p|_Gamma = 0 so that divu=0
            //     +innerProduct(p, 1./hi*pp*q)
            // , Kh, INTEGRAL_BOUNDARY);
            // Strong (p)
            Fun_h tmp_var(Wh, fun_exact_p);
            maxwell3D.setDirichletHone(tmp_var, Kh);

            // Both strong (u,p)
            // maxwell3D.setDirichletHoneAndHcurl(Wh, Uh, Kh);

            matlab::Export(maxwell3D.mat_[0], "mat" + std::to_string(i) + "Cut.dat");
            // continue;
            maxwell3D.solve("umfpack");

            // EXTRACT SOLUTION
            int nb_electric_dof = Uh.get_nb_dof();
            int nb_lagrange_dof = Wh.get_nb_dof();

            Rn_ data_uh = maxwell3D.rhs_(SubArray(nb_electric_dof, 0));
            Rn_ data_ph = maxwell3D.rhs_(SubArray(nb_lagrange_dof, nb_electric_dof)); // Rn_ data_uh = stokes.rhs_(SubArray(nb_vort_dof+nb_flux_dof,nb_vort_dof));

            Fun_h uh(Uh, data_uh);
            Fun_h ph(Wh, data_ph);

            auto uh_0dx = dx(uh.expr(0));
            auto uh_1dy = dy(uh.expr(1));
            auto uh_2dz = dz(uh.expr(2));

            // [Paraview]
            {
                Fun_h solu(Velh, fun_exact_u);

                Fun_h soluErr(Uh, fun_exact_u);
                soluErr.v -= uh.v;
                soluErr.v.map(fabs);

                // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

                Paraview<Mesh> writer(Kh, "maxwell_" + std::to_string(i) + ".vtk");
                writer.add(uh, "electric_field", 0, 3);
                writer.add(uh_0dx + uh_1dy + uh_2dz, "divergence");
                // writer.add(uh_0dx, "divergence");
                writer.add(solu, "electric_field_exact", 0, 3);
                writer.add(soluErr, "electric_field_error", 0, 3);
            }

            R errP      = L2norm(ph, fun_exact_p, 0, 1);
            R errU      = L2norm(uh, fun_exact_u, 0, 3);
            R errDiv    = L2norm(uh_0dx + uh_1dy + uh_2dz, fun_0, Kh);
            R maxErrDiv = maxNorm(uh_0dx + uh_1dy + uh_2dz, Kh);
            // R maxErrDiv = maxNorm(uh_0dx, Kh);

            h.push_back(hi);
            pl2.push_back(errP);
            ul2.push_back(errU);
            divl2.push_back(errDiv);
            divmax.push_back(maxErrDiv);
            if (i == 0) {
                convp.push_back(0);
                convu.push_back(0);
            } else {
                convp.push_back(log(pl2[i] / pl2[i - 1]) / log(h[i] / h[i - 1]));
                convu.push_back(log(ul2[i] / ul2[i - 1]) / log(h[i] / h[i - 1]));
            }
            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
        }
        std::cout << "\n"
        << std::left << std::setw(10) << std::setfill(' ') << "h" << std::setw(15) << std::setfill(' ')
        << "err p" << std::setw(15) << std::setfill(' ') << "conv p" << std::setw(15) << std::setfill(' ')
        << "err u" << std::setw(15) << std::setfill(' ') << "conv u" << std::setw(15) << std::setfill(' ')
        << "err divu"
        // << std::setw(15) << std::setfill(' ') << "conv divu"
        << std::setw(15) << std::setfill(' ')
        << "err maxdivu"
        // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
        << "\n"
        << std::endl;
        for (int i = 0; i < h.size(); ++i) {
            std::cout << std::left << std::setw(10) << std::setfill(' ') << h[i] << std::setw(15) << std::setfill(' ')
            << pl2[i] << std::setw(15) << std::setfill(' ') << convp[i] << std::setw(15) << std::setfill(' ')
            << ul2[i] << std::setw(15) << std::setfill(' ') << convu[i] << std::setw(15) << std::setfill(' ')
            << divl2[i]
            // << std::setw(15) << std::setfill(' ') << convdivPr[i]
            << std::setw(15) << std::setfill(' ')
            << divmax[i]
            // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
            << std::endl;
        }
    }
#endif

#ifdef UNFITTED_KIKUCHI

    using namespace globalVariable;
    namespace Data_Old {
        R k = 1.;
        R eps_r = 1.;
        R mu = 1.;

        R3 shift(0.5, 0.5, 0.5);

        R fun_levelSet(double *P, int i) {
            return (P[0] - shift.x) * (P[0] - shift.x) + (P[1] - shift.y) * (P[1] - shift.y) +
                (P[2] - shift.z) * (P[2] - shift.z) - 0.35 * 0.35 + Epsilon;
        }

        // R fun_rhs(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return 2 * pi * pi * sin(y * pi) * sin(z * pi) -
        //             eps_r * sin(y * pi) * sin(z * pi) * k * k; //+ 2 * (P[0] - shift.x) - eps_r * (2 * x - 1);
        //     else if (i == 1)
        //         return 2 * pi * pi * sin(x * pi) * sin(z * pi) -
        //             eps_r * sin(x * pi) * sin(z * pi) * k * k; //+ 2 * (P[1] - shift.y) - eps_r * (2 * y - 1);
        //     else
        //         return 2 * pi * pi * sin(x * pi) * sin(y * pi) -
        //             eps_r * sin(x * pi) * sin(y * pi) * k * k; //+ 2 * (P[2] - shift.z) - eps_r * (2 * z - 1);
        // }
        // R fun_exact_u(double *P, int i, int dom) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return sin(pi * y) * sin(pi * z);
        //     else if (i == 1)
        //         return sin(pi * x) * sin(pi * z);
        //     else
        //         return sin(pi * x) * sin(pi * y);
        // }
        // R fun_exact_curlu(double *P, int i, int dom) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return pi*(cos(pi*y) - cos(pi*z))*sin(pi*x);
        //     else if (i == 1)
        //         return pi*(-cos(pi*x) + cos(pi*z))*sin(pi*y);
        //     else
        //         return pi*(cos(pi*x) - cos(pi*y))*sin(pi*z);
        // }
        // R fun_exact_p(double *P, int i, int dom) {
        //     return (P[0] - shift.x) * (P[0] - shift.x) + (P[1] - shift.y) * (P[1] - shift.y) +
        //         (P[2] - shift.z) * (P[2] - shift.z) - 0.35 * 0.35;
        // }

        // Eriks example
        R fun_rhs(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return pi*pi*sin(pi*z) - sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_u(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_curlu(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return pi*cos(pi*z);
            else
                return 0;
        }

    }
    namespace Data_Cylinder {
        R fun_0(double *P, int i) {
            return 0;
        }

        R k = 2.;
        R eps = 1.;
        R mu = 1.;

        R cylrad = 0.2;
        R cylheight = 0.35; // half of cylinder height = 0.35
        R3 shift(0.0, 0.0, cylheight);
        R sdCylinder(double *p, float h, float r) {
            R px = p[0], py = p[1], pz = p[2];
            float length_p_xz = sqrt(px * px + pz * pz);
            float abs_p_y = fabs(py);
            float dx = fabs(length_p_xz) - r;
            float dy = fabs(abs_p_y) - h;
            
            float maxD = std::max(dx, dy);
            float minMaxD = std::min(maxD, 0.0f);
            
            float maxDx = std::max(dx, 0.0f);
            float maxDy = std::max(dy, 0.0f);
            float lengthMaxD = sqrt(maxDx * maxDx + maxDy * maxDy);
            
            return minMaxD + lengthMaxD;
        }
        R fun_levelSet(double *P, int i) {
            R3 pcyl(P[0]-shift.x,P[2]-shift.z,P[1]-shift.y); // cyl coords: (x,z,y)
            return sdCylinder(pcyl, cylheight, cylrad);
        }

        R x_0 = 2.405; // first zero of Bessel function J_0(x)
        R q0 = x_0/cylrad; // q0 = h of Cheng
        R ps = 11.8575134409; // propagation speed, sol to 4+x^2 = q0^2
        R fun_exact_u(double *P, int i, int dom) { // h^2 = k^2 + \gam^2, E = E_0 exp(-\gam z)
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0)  // -1/q0^2 ps dEz/dx
                return ps/q0 * x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1)
                return ps/q0 * y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return std::cyl_bessel_j(0, q0 * r)*exp(-ps*z);
        }
        R fun_exact_curlu(double *P, int i, int dom) { // curlF_theta = -dF/dz, e_theta = -x/r e_x + y/r e_y, J_0' = -J_1 
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0) // dy(Fz) - dz(Fy)
                return (-q0 + ps*ps/q0)*y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1) // dz(Fx) - dx(Fz)
                return (-ps*ps/q0 + q0)*x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return 0; // dx(Fy) - dy(Fx)
        }
        R fun_rhs(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_divu(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }
        R fun_exact_p(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }

    }
    using namespace Data_Cylinder;

    int main(int argc, char **argv) {
        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();
        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 7;
        int ny = 7;
        int nz = 7;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 3;
        for (int i = 0; i < iters; ++i) {
            R sh = M_PI*3e-2;
            Mesh3 Kh(nx, ny, nz, -0.2-sh, -0.2-sh, 0.0-sh, 0.4+sh, 0.4+sh, 2.01*cylheight+sh);
            const R hi = 1. / (nx - 1);

            Space Uh_(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            Space Wh_(Kh, DataFE<Mesh>::P1);

            Lagrange3 VelocitySpace(2);
            Space Vel_h(Kh, VelocitySpace);

            Space Lh(Kh, DataFE<Mesh>::P1);
            Fun_h levelSet(Lh, fun_levelSet);
            InterfaceLevelSet<Mesh> interface(Kh, levelSet);
            Normal n;

            // [Remove exterior]
            ActiveMesh<Mesh> Khi(Kh);
            Khi.truncate(interface, 1);
            Khi.info();

            CutSpace Velh(Khi, Vel_h);
            CutSpace Uh(Khi, Uh_);
            CutSpace Wh(Khi, Wh_);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h curlu0(Velh, fun_exact_curlu);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh);
            maxwell3D.add(Wh);

            /* Syntax:
            FunTest (fem space, #components, place in space)
            */
            FunTest u(Uh, 3, 0), v(Uh, 3, 0);
            FunTest p(Wh, 1, 0), q(Wh, 1, 0);

            // Eq 1
            maxwell3D.addBilinear( 
                +innerProduct(1./mu * 1./eps * curl(u), curl(v))
                -innerProduct(k * k * u, v)
                -innerProduct(grad(p), v)
                -innerProduct(u, grad(q))
            , Khi);
            maxwell3D.addLinear(
                +innerProduct(fh.exprList(), v)
            , Khi);
            R pp = 1e2;
            // maxwell3D.addBilinear( 
            //     -innerProduct(1./mu * curl(u), cross(n, v))
            //     -innerProduct(1./mu * cross(n, u), curl(v))
            //     +innerProduct(cross(n, u), 1./hi*pp*cross(n, v))
            // , interface);
            // maxwell3D.addLinear(
            //     -innerProduct(cross(n, u0), 1./mu * curl(v))
            //     +innerProduct(cross(n, u0), 1./hi*pp*cross(n, v))
            // , interface);
            maxwell3D.addBilinear( 
                +innerProduct(p, 1./hi*pp*q)
            , interface);
            maxwell3D.addLinear(
                +innerProduct(cross(n, curlu0), 1./mu * 1./eps * v)
            , interface);

            // [Stabilization]
            double tau_a = 1e0;
            double tau_b = 1e0;
            maxwell3D.addPatchStabilization(
            // W block
            + innerProduct(tau_a * pow(hi, 0) * jump(u), jump(v)) 
            // B blocks
            - innerProduct(tau_b * pow(hi, 0) * jump(grad(p)), jump(v))                     // -B^T block
            - innerProduct(tau_b * pow(hi, 0) * jump(u), jump(grad(q)))                     // B_0 block
            , Khi);

            matlab::Export(maxwell3D.mat_[0], "mat" + std::to_string(i) + "Cut.dat");
            maxwell3D.solve("umfpack");

            // EXTRACT SOLUTION

            int nb_electric_dof = Uh.get_nb_dof();
            int nb_lagrange_dof = Wh.get_nb_dof();

            Rn_ data_uh = maxwell3D.rhs_(SubArray(nb_electric_dof, 0));
            Rn_ data_ph = maxwell3D.rhs_(SubArray(nb_lagrange_dof, nb_electric_dof)); // Rn_ data_uh = stokes.rhs_(SubArray(nb_vort_dof+nb_flux_dof,nb_vort_dof));

            Fun_h uh(Uh, data_uh);
            Fun_h ph(Wh, data_ph);

            auto uh_0dx = dx(uh.expr(0));
            auto uh_1dy = dy(uh.expr(1));
            auto uh_2dz = dz(uh.expr(2));

            // [Paraview]
            {

                // Fun_h solw(Uh, fun_exact_w);

                Fun_h solu(Velh, fun_exact_u);
                Fun_h soluErr(Uh, fun_exact_u);
                Fun_h solp(Wh, fun_exact_p);

                soluErr.v -= uh.v;
                soluErr.v.map(fabs);

                // Fun_h divSolh(Wh, fun_div);
                // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

                Paraview<Mesh> writer(Khi, "maxwell_" + std::to_string(i) + ".vtk");

                writer.add(uh, "electric_field", 0, 3);
                writer.add(ph, "lag_mult", 0, 1);
                // writer.add(dx_uh0+dy_uh1, "divergence");
                // writer.add(femSol_0dx+femSol_1dy+fflambdah, "divergence");
                // writer.add(solp, "pressureExact", 0, 1);
                writer.add(solu, "electric_field_exact", 0, 3);
                writer.add(soluErr, "electric_field_error", 0, 3);

                R errU           = L2normCut(uh, fun_exact_u, 0, 3);
                // R errP           = 0;
                R errP           = L2normCut(ph, fun_exact_p, 0, 1);
                double errDiv    = L2normCut(uh_0dx + uh_1dy + uh_2dz, Khi);
                double maxErrDiv = maxNormCut(uh_0dx + uh_1dy + uh_2dz, Khi);

                ul2.push_back(errU);
                pl2.push_back(errP);
                divl2.push_back(errDiv);
                divmax.push_back(maxErrDiv);
                h.push_back(hi);
                if (i == 0) {
                    convu.push_back(0);
                    convp.push_back(0);
                } else {
                    convu.push_back(log(ul2[i] / ul2[i - 1]) / log(h[i] / h[i - 1]));
                    convp.push_back(log(pl2[i] / pl2[i - 1]) / log(h[i] / h[i - 1]));
                }
                nx = 2 * nx - 1;
                ny = 2 * ny - 1;
                nz = 2 * nz - 1;
            }
        }
        std::cout << "\n"
        << std::left << std::setw(10) << std::setfill(' ') << "h" << std::setw(15) << std::setfill(' ')
        << "err p" << std::setw(15) << std::setfill(' ') << "conv p" << std::setw(15) << std::setfill(' ')
        << "err u" << std::setw(15) << std::setfill(' ') << "conv u" << std::setw(15) << std::setfill(' ')
        << "err divu"
        // << std::setw(15) << std::setfill(' ') << "conv divu"
        << std::setw(15) << std::setfill(' ')
        << "err maxdivu"
        // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
        << "\n"
        << std::endl;
        for (int i = 0; i < h.size(); ++i) {
            std::cout << std::left << std::setw(10) << std::setfill(' ') << h[i] << std::setw(15) << std::setfill(' ')
            << pl2[i] << std::setw(15) << std::setfill(' ') << convp[i] << std::setw(15) << std::setfill(' ')
            << ul2[i] << std::setw(15) << std::setfill(' ') << convu[i] << std::setw(15) << std::setfill(' ')
            << divl2[i]
            // << std::setw(15) << std::setfill(' ') << convdivPr[i]
            << std::setw(15) << std::setfill(' ')
            << divmax[i]
            // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
            << std::endl;
        }
    }

#endif


#ifdef FITTED_3FIELD_EIGEN
    bool usingLagrangeMultiplierBC = false;

    using namespace globalVariable;
    namespace Data_mueps { // f = 1/eps curl j => div f = 0 !
        R k = 1.;
        R eps = 1.;
        R mu = 1.;

        // For the boundary mesh
        double shift = 0.5*M_PI;
        // R sdRoundBox( double* p, double* b, double r ) {
        //     R qx = std::abs(p[0]) - b[0] + r;
        //     R qy = std::abs(p[1]) - b[1] + r;
        //     R qz = std::abs(p[2]) - b[2] + r;

        //     R val = sqrt(pow(std::max(qx,0.0),2)+pow(std::max(qy,0.0),2)+pow(std::max(qz,0.0),2)) + std::min(std::max(qx,std::max(qy,qz)),0.0) - r;
        //     // std::cout << val << std::endl;
        //     return val;
        // }
        // R fun_levelSet(double *P, int i, int dom) {
        //     double Pnew[3] = {P[0]-shift, P[1]-shift, P[2]-shift};

        //     // half-dimensions of box 
        //     double sh_int = 1e-12; // M_PI/6+1e-12;
        //     double b[3] = {0.5*M_PI - sh_int, 0.5*M_PI - sh_int, 0.5*M_PI - sh_int}; // if using GMSH mesh

        //     // smoothing radius
        //     double r = 0.025;
        //     return sdRoundBox(Pnew, b, r);
        // }
        // R fun_levelSet(double *P_, int i, int dom) {
        //     double P[3] = {P_[0]-shift, P_[1]-shift, P_[2]-shift};
        //     R x = P[0], y = P[1], z = P[2];
        //     return pow(x,2) + pow(y,2) + pow(z,2) - (pow(0.5*M_PI,2)-1e-12);
        // }
        // R fun_levelSet(double *P, int i, int dom) {
        //     R x = P[0], y = P[1], z = P[2];
        //     R F = 0.5*M_PI/6+1e-12;
        //     R T = M_PI-0.5*M_PI/6-1e-12;
        //     if (x>F && x<T && y>F && y<T && z>F && z<T)
        //         return -1;
        //     else
        //         return 1;
        // }
        R fun_levelSet(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            R F = 1e-12;//0.5*M_PI/6+1e-12;
            R T = M_PI-1e-12; //M_PI-0.5*M_PI/6-1e-12;
            return (F-x)*(x-T)*(F-y)*(y-T)*(F-z)*(z-T);
        }

        // Eriks example
        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return pi*pi*sin(pi*z) - sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_u(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_curlu(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return pi*cos(pi*z);
            else
                return 0;
        }
        R fun_exact_p(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return x*(x-1)-y*(y-1);
        }

        R fun_closed_form(double *P, int i) { // = grad 1/r
            R x = P[0], y = P[1], z = P[2];
            R r = sqrt(x*x + y*y + z*z);
            R r3 = r*r*r;
            if (i == 0)
                return x/r3;
            else if (i == 1)
                return y/r3;
            else
                return z/r3;
        }

        R fun_0(double *P, int i) {
            return 0;
        }

    }

    using namespace Data_mueps;
    int main(int argc, char **argv) {

        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();

        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 7;
        int ny = 7;
        int nz = 7;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 2;
        for (int i = 0; i < iters; ++i) {
            // Mesh3 Kh_(nx, ny, nz, 0., 0., 0., M_PI, M_PI, M_PI);
            // Mesh3 Kh_("../cpp/mainFiles/meshes/cube_"+std::to_string(i), MeshFormat::mesh_gmsh);
            // Mesh3 Kh_("../cpp/mainFiles/meshes/cube_hole_"+std::to_string(i), MeshFormat::mesh_gmsh);
            Mesh3 Kh_("../cpp/mainFiles/meshes/ball_hole_"+std::to_string(i), MeshFormat::mesh_gmsh);
            Kh_.info();
            const R hi = 1. / (nx - 1); // 1./(nx-1)

            Space Uh_(Kh_, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            Space Vh_(Kh_, DataFE<Mesh>::RT0);
            Space Wh_(Kh_, DataFE<Mesh>::P0);
            Lagrange3 VelocitySpace(2); Space Velh_(Kh_, VelocitySpace);

            // Create ActiveMesh and CutSpaces
            ActiveMesh<Mesh> Kh(Kh_);
            CutSpace Uh(Kh, Uh_);
            CutSpace Vh(Kh, Vh_);
            CutSpace Wh(Kh, Wh_);
            CutSpace Velh(Kh, Velh_);

            // Paraview
            Space Lh_(Kh_, DataFE<Mesh>::P1); CutSpace Lh(Kh, Lh_);
            Fun_h levelSet(Lh, fun_levelSet);
            Paraview<Mesh> writer(Kh, "maxwell_" + std::to_string(i) + ".vtk");
            writer.add(levelSet, "levelset", 0, 1);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h not_exact_form(Velh, fun_closed_form);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh); maxwell3D.add(Vh); maxwell3D.add(Wh);
            CutFEM<Mesh> massRHS(Uh); massRHS.add(Vh); massRHS.add(Wh);
            int nb_dof = Uh.get_nb_dof()+Vh.get_nb_dof()+Wh.get_nb_dof();

            Normal n;
            /* Syntax:
            FunTest (fem space, #components, place in space)
            */
            FunTest w(Uh, 3, 0), tau(Uh, 3, 0);
            FunTest u(Vh, 3, 0), v(Vh, 3, 0), p(Wh, 1, 0), q(Wh, 1, 0);

            // [Bulk]
            // Eq 1
            maxwell3D.addBilinear( // w = curl u
                -innerProduct(eps * mu * w, tau) 
                +innerProduct(u, curl(tau))
            , Kh);
            // Eq 2
            maxwell3D.addBilinear( // mu Delta u + grad p
                +innerProduct(curl(w), v)
                // -innerProduct(k * k * u, v)
                +innerProduct(p, div(v))
            , Kh);
            // Eq 3
            maxwell3D.addBilinear(
                +innerProduct(div(u), q)
            , Kh);

            // » IF using cube with hole mesh
            // maxwell3D.addLagrangeMultiplier(
            //     +innerProduct(not_exact_form.exprList(), v), 0
            // , Kh);
            // maxwell3D.mat_[0][std::make_pair(nb_dof,nb_dof)] = 0; // For PETsc
            // // FEM<Mesh> lagr(Uh); lagr.add(Vh); lagr.add(Wh);
            // // lagr.addLinear(innerProduct(not_exact_form.exprList(), u), Kh);
            // // Rn lag_row(lagr.rhs_); 
            // // lagr.rhs_ = 0.; 
            // // lagr.addLinear(innerProduct(not_exact_form.exprList(), v), Kh);
            // // maxwell3D.addLagrangeVecToRowAndCol(lag_row, lagr.rhs_, 0);
            // massRHS.addLagrangeMultiplier(
            //     -innerProduct(not_exact_form.exprList(), 0*v), 0
            // , Kh);
            // massRHS.mat_[0][std::make_pair(nb_dof,nb_dof)] = 0; // For PETsc

            if (!usingLagrangeMultiplierBC) {
                R regularizer = 1e-12/(hi*hi*hi); // 1e-12
                // » For PETSc
                maxwell3D.addBilinear(
                    +innerProduct(0*u,v)
                    +innerProduct(0*p,q)
                , Kh);

                // » Essential BC Nitsche
                // R pp = 1e2;
                // maxwell3D.addBilinear(
                //     +innerProduct(u, cross(n,tau))
                //     +innerProduct(cross(n,w), v)
                //     -innerProduct(cross(n,w), pp*1./hi * cross(n,tau))
                // , Kh, INTEGRAL_BOUNDARY);
                // » Essential BC strong
                Fun_h fun0(Uh, fun_0);
                maxwell3D.setDirichletHcurl(fun0, Kh);

                // » RHS MAT
                massRHS.addBilinear( 
                    +innerProduct(w, 0*tau) // 0*
                    +innerProduct(u, v)
                    +innerProduct(p, regularizer*q)
                , Kh);
                massRHS.setDirichletHcurl_RHSMat(fun0, Kh);

                // » If want zero pressure average via Lagrange multiplier
                // maxwell3D.addLagrangeMultiplier(
                //     +innerProduct(1, p), 0
                // , Kh);
                // CutFEM<Mesh> lagr(Uh); lagr.add(Vh); lagr.add(Wh);
                // lagr.addLinear(innerProduct(1, p), Kh);
                // Rn lag_row(lagr.rhs_); 
                // lagr.rhs_ = 0.; 
                // lagr.addLinear(innerProduct(1, v*n), Kh, INTEGRAL_BOUNDARY);
                // maxwell3D.addLagrangeVecToRowAndCol(lag_row, lagr.rhs_, 0);
                // massRHS.mat_[0][std::make_pair(nb_dof,nb_dof)] = 0; // For PETsc
            } else { // Lagrange multiplier for boundary condition
                ActiveMesh<Mesh> Khsurf(Kh_); // Remove interior
                // InterfaceLevelSet<Mesh> interface(Kh_, levelSet); Normal n;
                // Khsurf.createSurfaceMesh(interface);
                Khsurf.createBoundaryMesh();
                Khsurf.info();

                // Paraview
                Paraview<Mesh> writer(Khsurf, "maxwell_bdrymesh_" + std::to_string(i) + ".vtk");
                writer.add(levelSet, "levelset", 0, 1);

                LagrangeDC3 P0dc3(1); Space WWWh(Kh_, P0dc3); CutSpace WWWh_itf(Khsurf, WWWh);
                // Lagrange3 P13(1); Space WWWh(Kh_, P13); CutSpace WWWh_itf(Khsurf, WWWh);
                maxwell3D.add(WWWh_itf);
                FunTest p_itf(WWWh_itf, 3, 0), q_itf(WWWh_itf, 3, 0);
                maxwell3D.addBilinear(
                    +innerProduct(p_itf, cross(n,tau))
                    +innerProduct(cross(n,w), q_itf)
                    // +innerProduct(p_itf, q_itf)
                , Kh_, INTEGRAL_BOUNDARY);
                // For PETSc
                maxwell3D.addBilinear(
                    +innerProduct(0*u,v)
                    +innerProduct(0*p,q)
                , Kh);
                maxwell3D.addBilinear(
                    +innerProduct(0*p_itf,q_itf)
                , Khsurf);

                // RHS MAT B 
                R regularizer = 1e-12/(hi*hi*hi); // divide by vol
                massRHS.addBilinear( 
                    +innerProduct(u, v)
                    +innerProduct(w, regularizer*tau)
                    +innerProduct(p, regularizer*q)
                , Kh);
                massRHS.add(WWWh_itf);
                massRHS.addBilinear(
                    +innerProduct(p_itf, regularizer*q_itf)
                , Khsurf);

                std::cout << "3d P0dc dofs:" << WWWh_itf.get_nb_dof() << std::endl;    
            }

            matlab::Export(maxwell3D.mat_[0], "A" + std::to_string(i) + ".dat");
            matlab::Export(massRHS.mat_[0], "B" + std::to_string(i) + ".dat");

            std::cout << Uh.get_nb_dof() << std::endl;
            std::cout << Vh.get_nb_dof() << std::endl;
            std::cout << Wh.get_nb_dof() << std::endl;

            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
            continue;
        }
    }
#endif

#ifdef UNFITTED_3FIELD_EIGEN

    bool usingLagrangeMultiplierBC = false;

    using namespace globalVariable;
    namespace Data_CubeLevelset { // f = 1/eps curl j => div f = 0 !
        R k = 1.;
        R eps = 1.;
        R mu = 1.;

        double shift = 0.5*M_PI;
        R sdRoundBox( double* p, double* b, double r ) {
            R qx = std::abs(p[0]) - b[0] + r;
            R qy = std::abs(p[1]) - b[1] + r;
            R qz = std::abs(p[2]) - b[2] + r;

            R val = sqrt(pow(std::max(qx,0.0),2)+pow(std::max(qy,0.0),2)+pow(std::max(qz,0.0),2)) + std::min(std::max(qx,std::max(qy,qz)),0.0) - r;
            // std::cout << val << std::endl;
            return val;
        }
        R fun_levelSet(double *P, int i, int dom) {
            double Pnew[3] = {P[0]-shift, P[1]-shift, P[2]-shift};

            // half-dimensions of box 
            // double b[3] = {0.5*M_PI, 0.5*M_PI, 0.5*M_PI}; // if using cutfem mesh
            double sh_int = 0.1+1e-12;
            double b[3] = {0.5*M_PI - sh_int, 0.5*M_PI - sh_int, 0.5*M_PI - sh_int}; // if using GMSH mesh

            // smoothing radius
            double r = 0.025;
            return sdRoundBox(Pnew, b, 0.4);
        }
        R fun_levelSetPLANE(double *P, int i, int dom) {
            return P[2] - M_PI; // was 1e-1..
        }

        // Eriks example
        R fun_rhs(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return pi*pi*sin(pi*z) - sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_u(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_curlu(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return pi*cos(pi*z);
            else
                return 0;
        }
        R fun_exact_p(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            return x*(x-1)-y*(y-1);
        }

        R fun_closed_form(double *P, int i, int dom) { // = grad 1/r
            R x = P[0], y = P[1], z = P[2];
            R r = sqrt(x*x + y*y + z*z);
            R r3 = r*r*r;
            if (i == 0)
                return x/r3;
            else if (i == 1)
                return y/r3;
            else
                return z/r3;
        }

        R fun_0(double *P, int i, int dom) {
            return 0;
        }

    }

    using namespace Data_CubeLevelset;
    int main(int argc, char **argv) {

        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();

        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 7;
        int ny = 7;
        int nz = 7;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 2;
        for (int i = 0; i < iters; ++i) {
            // R sh = 0.234225;
            Mesh3 Kh(nx, ny, nz, 0, 0, 0, M_PI, M_PI, M_PI+1e-12); // see Mesh3dn.hpp
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_"+std::to_string(i), MeshFormat::mesh_gmsh);
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_hole_"+std::to_string(i), MeshFormat::mesh_gmsh);
            Kh.info();
            const R hi = 1. / (nx - 1); // 1./(nx-1)

            Space Uh_(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            Space Vh_(Kh, DataFE<Mesh>::RT0);
            Space Wh_(Kh, DataFE<Mesh>::P0);

            Lagrange3 VelocitySpace(2);
            Space Vel_h(Kh, VelocitySpace);

            Space Lh(Kh, DataFE<Mesh>::P1);
            // Fun_h levelSet(Lh, fun_levelSet);
            Fun_h levelSet(Lh, fun_levelSetPLANE);
            InterfaceLevelSet<Mesh> interface(Kh, levelSet);
            Normal n;

            // [Remove exterior]
            ActiveMesh<Mesh> Khi(Kh);
            Khi.truncate(interface, 1);
            Khi.info();

            CutSpace Velh(Khi, Vel_h);
            CutSpace Uh(Khi, Uh_);
            CutSpace Vh(Khi, Vh_);
            CutSpace Wh(Khi, Wh_);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h not_exact_form(Velh, fun_closed_form);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh); maxwell3D.add(Vh); maxwell3D.add(Wh);

            /* Syntax:
            FunTest (fem space, #components, place in space)
            */
            FunTest w(Uh, 3, 0), tau(Uh, 3, 0);
            FunTest u(Vh, 3, 0), v(Vh, 3, 0), p(Wh, 1, 0), q(Wh, 1, 0);

            // std::getchar();
            Fun_h fun0(Wh, fun_0);
            Paraview<Mesh> writer(Khi, "maxwell_" + std::to_string(i) + ".vtk");
            writer.add(fun0, "vol", 0, 1);

            // [Bulk]
            // Eq 1
            maxwell3D.addBilinear( // w = curl u
                -innerProduct(eps * mu * w, tau) 
                +innerProduct(u, curl(tau))
            , Khi);
            // Eq 2
            maxwell3D.addBilinear( // mu Delta u + grad p
                +innerProduct(curl(w), v)
                // -innerProduct(k * k * u, v)
                +innerProduct(p, div(v))
            , Khi);
            // Eq 3
            maxwell3D.addBilinear(
                +innerProduct(div(u), q)
            , Khi);
            // [Stabilization]
            double tau_w = 1e0;
            double tau_m = 1e0;
            double tau_a = 1e0;
            double tau_b = 1e0;
            maxwell3D.addPatchStabilization(
                -innerProduct(tau_w * jump(w), jump(tau)) 
                +innerProduct(tau_m * jump(curl(w)), jump(v))     
                +innerProduct(tau_m * jump(u), jump(curl(tau)))    
                +innerProduct(tau_b * jump(p), jump(div(v)))     
                +innerProduct(tau_b * jump(div(u)), jump(q))    
            , Khi);

            if (!usingLagrangeMultiplierBC) {
                // Essential BC (natural BC has no matrix terms)
                R pp = 1e2;
                // maxwell3D.addBilinear(
                //     +innerProduct(u, cross(n,tau))
                //     -innerProduct(cross(n,w), v)
                //     +innerProduct(cross(n,w), pp*1./hi * cross(n,tau))
                // , interface);
                // maxwell3D.addBilinear(
                //     +innerProduct(u, cross(n,tau))
                //     -innerProduct(cross(n,w), v)
                //     +innerProduct(cross(n,w), pp*1./hi * cross(n,tau))
                // , Khi, INTEGRAL_BOUNDARY);
            } else {
                // Lagrange multiplier for boundary condition
                ActiveMesh<Mesh> Kh_itf(Kh);
                Kh_itf.createBoundaryAndSurfaceMesh(interface);
                CutSpace Wh_itf(Kh_itf, Wh_);   
                maxwell3D.add(Wh_itf);
                FunTest p_itf(Wh_itf, 1, 0), q_itf(Wh_itf, 1, 0);
                maxwell3D.addBilinear(
                    +innerProduct(p_itf, v*n)
                    +innerProduct(u*n, q_itf)
                    , interface
                );
                maxwell3D.addBilinear(
                    +innerProduct(p_itf, v*n)
                    +innerProduct(u*n, q_itf)
                    , Kh_itf, INTEGRAL_BOUNDARY
                );
            }

            matlab::Export(maxwell3D.mat_[0], "A" + std::to_string(i) + ".dat");

            // Eigenvalue problem
            CutFEM<Mesh> massRHS(Uh); massRHS.add(Vh); massRHS.add(Wh);
            R el_area = hi*hi*hi;
            R regularizer = 1e-12/el_area;
            massRHS.addBilinear( 
                +innerProduct(u, v)
                +innerProduct(w, 0*tau)
                +innerProduct(p, regularizer*q)
            , Khi);
            massRHS.addPatchStabilization(
                +innerProduct(tau_m * jump(u), jump(v))
            , Khi);

            matlab::Export(massRHS.mat_[0], "B" + std::to_string(i) + ".dat");
            
            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
            continue;
        }
    }
#endif

#ifdef FITTED_3FIELD

    using namespace globalVariable;
    namespace Data_CubeHole { // f = 1/eps curl j => div f = 0 !
        R k = 1.;
        R eps = 1.;
        R mu = 1.;

        // Monks example
        // R fun_rhs(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return pi*pi*sin(pi*z) - sin(pi*z);
        //     else if (i == 1)
        //         return 0;
        //     else
        //         return 0;
        // }
        // R fun_exact_u(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     R cA = 1./sqrt(7./2);
        //     R ck = 1./(sqrt(13)/2);
        //     if (i == 0)
        //         return cA*cos(ck*(-3./2*x+y));
        //     else if (i == 1)
        //         return cA*3./2*cos(ck*(-3./2*x+y));
        //     else
        //         return cA*1./2*cos(ck*(-3./2*x+y));
        // }

        // Eriks example
        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return pi*pi*sin(pi*z) - sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_u(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi*z);
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_curlu(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return pi*cos(pi*z);
            else
                return 0;
        }
        R fun_exact_p(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return x*(x-1)-y*(y-1);
        }

        R fun_closed_form(double *P, int i) { // = grad 1/r
            R x = P[0], y = P[1], z = P[2];
            R r = sqrt(x*x + y*y + z*z);
            R r3 = r*r*r;
            if (i == 0)
                return x/r3;
            else if (i == 1)
                return y/r3;
            else
                return z/r3;
        }

        R fun_0(double *P, int i) {
            return 0;
        }

    }

    namespace Data_Cylinder { // f = 1/eps curl j => div f = 0 !
        R fun_0(double *P, int i) {
            return 0;
        }

        R k = 2.;
        R eps = 1.;
        R mu = 1.;
    

        R cylrad = 0.2;

        // Radially outward Efield 
        // R fun_exact_u(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     R r2 = x*x + y*y;
        //     if (i == 0)
        //         return 1./(3*eps)*r2;
        //     else if (i == 1)
        //         return 1./(3*eps)*r2;
        //     else
        //         return 0;
        // }
        // R fun_rhs(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return -2./(3*eps)-k*k*fun_exact_u(P,0);
        //     else if (i == 1)
        //         return -2./(3*eps)-k*k*fun_exact_u(P,1);
        //     else
        //         return 0;
        // }
        // R fun_exact_curlu(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     if (i == 0)
        //         return 0;
        //     else if (i == 1)
        //         return 0;
        //     else
        //         return 2./(3*eps)*(x-y);
        // }
        // R fun_exact_divu(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     return 2./(3*eps)*(x+y);
        // }
        // R fun_exact_p(double *P, int i) {
        //     R x = P[0], y = P[1], z = P[2];
        //     return 0;
        // }

        R x_0 = 2.405; // first zero of Bessel function J_0(x)
        R q0 = x_0/cylrad; // q0 = h of Cheng
        R ps = 11.8575134409; // propagation speed, sol to 4+x^2 = q0^2
        R fun_exact_u(double *P, int i) { // h^2 = k^2 + \gam^2, E = E_0 exp(-\gam z)
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0)  // -1/q0^2 ps dEz/dx
                return ps/q0 * x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1)
                return ps/q0 * y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return std::cyl_bessel_j(0, q0 * r)*exp(-ps*z);
        }
        R fun_exact_curlu(double *P, int i) { // curlF_theta = -dF/dz, e_theta = -x/r e_x + y/r e_y, J_0' = -J_1 
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0) // dy(Fz) - dz(Fy)
                return (-q0 + ps*ps/q0)*y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1) // dz(Fx) - dx(Fz)
                return (-ps*ps/q0 + q0)*x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return 0; // dx(Fy) - dy(Fx)
        }
        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_divu(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }
        R fun_exact_p(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }

    }

    using namespace Data_Cylinder;

    int main(int argc, char **argv) {

        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();

        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 5;
        int ny = 5;
        int nz = 5;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 2;
        for (int i = 0; i < iters; ++i) {
            // Mesh3 Kh(nx, ny, nz, 0., 0., 0., 1., 1., 1.);
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_"+std::to_string(i), MeshFormat::mesh_gmsh);
            // Mesh3 Kh("../cpp/mainFiles/meshes/cube_hole_"+std::to_string(i), MeshFormat::mesh_gmsh);
            Mesh3 Kh("../cpp/mainFiles/meshes/cyli_"+std::to_string(i), MeshFormat::mesh_gmsh);
            // Kh.info();
            const R hi = 1. / (nx - 1); // 1./(nx-1)

            Space Uh(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            Space Vh(Kh, DataFE<Mesh>::RT0);
            Space Wh(Kh, DataFE<Mesh>::P0);

            Lagrange3 VelocitySpace(2);
            Space Velh(Kh, VelocitySpace);
            Space Scalh(Kh, DataFE<Mesh>::P1);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h gh(Scalh, fun_exact_divu);
            // Fun_h not_exact_form(Velh, fun_closed_form);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh); maxwell3D.add(Vh); maxwell3D.add(Wh);

            Normal n;
            /* Syntax:
            FunTest (fem space, #components, place in space)
            */
            FunTest w(Uh, 3, 0), tau(Uh, 3, 0);
            FunTest u(Vh, 3, 0), v(Vh, 3, 0), p(Wh, 1, 0), q(Wh, 1, 0);

            // [Bulk]
            // Eq 1
            maxwell3D.addBilinear( // w = curl u
                -innerProduct(mu * eps * w, tau) 
                +innerProduct(u, curl(tau))
            , Kh);
            // Eq 2
            maxwell3D.addBilinear( // mu Delta u + grad p
                +innerProduct(curl(w), v)
                -innerProduct(k * k * u, v)
                +innerProduct(p, div(v))
            , Kh);
            maxwell3D.addLinear(
                +innerProduct(fh.exprList(), 1./eps * v)
            , Kh);
            // Eq 3
            maxwell3D.addBilinear(
                +innerProduct(div(u), q)
            , Kh);
            maxwell3D.addLinear(
                +innerProduct(gh.expr(), q)
            , Kh);
            // IF using cube with hole mesh
            // maxwell3D.addLagrangeMultiplier(
            //     +innerProduct(not_exact_form.exprList(), u), 0
            // , Kh);
            // FEM<Mesh> lagr(Uh); lagr.add(Vh); lagr.add(Wh);
            // lagr.addLinear(innerProduct(not_exact_form.exprList(), u), Kh);
            // Rn lag_row(lagr.rhs_); 
            // lagr.rhs_ = 0.; 
            // lagr.addLinear(innerProduct(not_exact_form.exprList(), v), Kh);
            // maxwell3D.addLagrangeVecToRowAndCol(lag_row, lagr.rhs_, 0);

            // » Essential BC Nitsche
            // R pp = 1e2;
            // maxwell3D.addBilinear(
            //     +innerProduct(u, cross(n,tau))
            //     +innerProduct(cross(n,w), v)
            //     -innerProduct(cross(n,w), pp*1./hi * cross(n,tau))
            // , Kh, INTEGRAL_BOUNDARY);
            // maxwell3D.addLinear(
            //     -innerProduct(cross(n, u0), tau)
            // , Kh, INTEGRAL_BOUNDARY);
            // » Essential BC strong
            Fun_h fun0(Uh, fun_0);
            maxwell3D.setDirichletHcurl(fun0, Kh);

            matlab::Export(maxwell3D.mat_[0], "mat" + std::to_string(i) + ".dat");
            maxwell3D.solve("umfpack");
            
            // EXTRACT SOLUTION
            int nb_vort_dof = Uh.get_nb_dof();
            int nb_flux_dof = Vh.get_nb_dof();

            Rn_ data_wh = maxwell3D.rhs_(SubArray(nb_vort_dof, 0));
            Rn_ data_uh = maxwell3D.rhs_(SubArray(
                nb_flux_dof, nb_vort_dof)); // Rn_ data_uh = stokes.rhs_(SubArray(nb_vort_dof+nb_flux_dof,nb_vort_dof));
            Rn_ data_ph = maxwell3D.rhs_(SubArray(Wh.get_nb_dof(), nb_vort_dof + nb_flux_dof)); //

            Fun_h wh(Uh, data_wh);
            Fun_h uh(Vh, data_uh);
            Fun_h ph(Wh, data_ph);

            auto uh_0dx = dx(uh.expr(0));
            auto uh_1dy = dy(uh.expr(1));
            auto uh_2dz = dz(uh.expr(2));

            // [Paraview]
            {
                // Fun_h solw(Uh, fun_exact_w);

                Fun_h solu(Velh, fun_exact_u);
                Fun_h soluErr(Vh, fun_exact_u);
                Fun_h solp(Wh, fun_exact_p);

                soluErr.v -= uh.v;
                soluErr.v.map(fabs);

                // Fun_h divSolh(Wh, fun_div);
                // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

                Paraview<Mesh> writer(Kh, "maxwell_" + std::to_string(i) + ".vtk");

                writer.add(wh, "vorticity", 0, 3);
                writer.add(uh, "velocity", 0, 3);
                writer.add(ph, "pressure", 0, 1);

                // writer.add(dx_uh0+dy_uh1, "divergence");
                // writer.add(femSol_0dx+femSol_1dy+fflambdah, "divergence");

                writer.add(solp, "pressureExact", 0, 1);
                writer.add(solu, "velocityExact", 0, 2);
                writer.add(soluErr, "velocityError", 0, 2);
            }

            R errU      = L2norm(uh, fun_exact_u, 0, 3);
            R errP      = L2norm(ph, fun_exact_p, 0, 1);
            R errDiv    = L2norm(uh_0dx + uh_1dy + uh_2dz, fun_0, Kh);
            R maxErrDiv = maxNorm(uh_0dx + uh_1dy + uh_2dz, Kh);

            // Face jump errors as measure of weak divergence error
            Space P0_(Kh, DataFE<Mesh>::P0); FunTest q0(P0_, 1, 0);
            CutFEM<Mesh> err_face_jumps(P0_);
            err_face_jumps.BaseFEM<Mesh>::addLinearSquareIntegrand(
                // +innerProduct(1, jump(v*n))
                // +innerProduct(1, jump(q0))
                +innerProduct(1, jump(uh*n * q0))
                // +innerProduct(1, jump(wh*n * q0))
            , Kh, INTEGRAL_INNER_FACE_3D);
            errDiv = sqrt(abs(err_face_jumps.rhs_.sum()));
            // errDiv = abs(err_face_jumps.rhs_.sum());

            ul2.push_back(errU);
            pl2.push_back(errP);
            divl2.push_back(errDiv);
            divmax.push_back(maxErrDiv);
            h.push_back(hi);
            if (i == 0) {
                convu.push_back(0);
                convp.push_back(0);
            } else {
                convu.push_back(log(ul2[i] / ul2[i - 1]) / log(h[i] / h[i - 1]));
                convp.push_back(log(pl2[i] / pl2[i - 1]) / log(h[i] / h[i - 1]));
            }
            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
        }
        std::cout << "\n"
        << std::left << std::setw(10) << std::setfill(' ') << "h" << std::setw(15) << std::setfill(' ')
        << "err p" << std::setw(15) << std::setfill(' ') << "conv p" << std::setw(15) << std::setfill(' ')
        << "err u" << std::setw(15) << std::setfill(' ') << "conv u" << std::setw(15) << std::setfill(' ')
        << "err divu"
        // << std::setw(15) << std::setfill(' ') << "conv divu"
        // << std::setw(15) << std::setfill(' ') << "err_new divu"
        // << std::setw(15) << std::setfill(' ') << "convLoc divu"
        << std::setw(15) << std::setfill(' ')
        << "err maxdivu"
        // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
        << "\n"
        << std::endl;
        for (int i = 0; i < h.size(); ++i) {
            std::cout << std::left << std::setw(10) << std::setfill(' ') << h[i] << std::setw(15) << std::setfill(' ')
            << pl2[i] << std::setw(15) << std::setfill(' ') << convp[i] << std::setw(15) << std::setfill(' ')
            << ul2[i] << std::setw(15) << std::setfill(' ') << convu[i] << std::setw(15) << std::setfill(' ')
            << divl2[i]
            // << std::setw(15) << std::setfill(' ') << convdivPr[i]
            // << std::setw(15) << std::setfill(' ') << divPrintLoc[i]
            // << std::setw(15) << std::setfill(' ') << convdivPrLoc[i]
            << std::setw(15) << std::setfill(' ')
            << divmax[i]
            // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
            << std::endl;
        }
    }
#endif

#ifdef UNFITTED_3FIELD

    using namespace globalVariable;
    namespace Data_Sphere {
        R k = 1.;
        R eps_r = 1.;

        R3 shift(0.5, 0.5, 0.5);

        R fun_levelSet(double *P, int i) {
            return (P[0] - shift.x) * (P[0] - shift.x) + (P[1] - shift.y) * (P[1] - shift.y) +
                (P[2] - shift.z) * (P[2] - shift.z) - 0.35 * 0.35 + Epsilon;
        }

        R fun_rhs(double *P, int i) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 2 * pi * pi * sin(y * pi) * sin(z * pi) - eps_r * (2 * x - 1) -
                    eps_r * sin(y * pi) * sin(z * pi) * k * k;
            else if (i == 1)
                return 2 * pi * pi * sin(x * pi) * sin(z * pi) - eps_r * (2 * y - 1) -
                    eps_r * sin(x * pi) * sin(z * pi) * k * k;
            else
                return 2 * pi * pi * sin(x * pi) * sin(y * pi) - eps_r * (2 * z - 1) -
                    eps_r * sin(x * pi) * sin(y * pi) * k * k;
        }
        // R fun_boundary(double *P, int i) {
        //     if (i == 0)
        //         return 0.;
        //     else if ()
        // }
        R fun_exact_u(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return sin(pi * y) * sin(pi * z);
            else if (i == 1)
                return sin(pi * x) * sin(pi * z);
            else
                return sin(pi * x) * sin(pi * y);
        }
        R fun_exact_p(double *P, int i, int dom) {
            return (P[0] - shift.x) * (P[0] - shift.x) + (P[1] - shift.y) * (P[1] - shift.y) +
                (P[2] - shift.z) * (P[2] - shift.z) - 0.35 * 0.35;
        }
    } 
    namespace Data_Cylinder {
        R fun_0(double *P, int i) {
            return 0;
        }

        R k = 2.;
        R eps = 1.;
        R mu = 1.;

        R cylrad = 0.2;
        R cylheight = 0.35; // half of cylinder height = 0.35
        R3 shift(0.0, 0.0, cylheight);
        R sdCylinder(double *p, float h, float r) {
            R px = p[0], py = p[1], pz = p[2];
            float length_p_xz = sqrt(px * px + pz * pz);
            float abs_p_y = fabs(py);
            float dx = fabs(length_p_xz) - r;
            float dy = fabs(abs_p_y) - h;
            
            float maxD = std::max(dx, dy);
            float minMaxD = std::min(maxD, 0.0f);
            
            float maxDx = std::max(dx, 0.0f);
            float maxDy = std::max(dy, 0.0f);
            float lengthMaxD = sqrt(maxDx * maxDx + maxDy * maxDy);
            
            return minMaxD + lengthMaxD;
        }
        R fun_levelSet(double *P, int i) {
            R3 pcyl(P[0]-shift.x,P[2]-shift.z,P[1]-shift.y); // cyl coords: (x,z,y)
            return sdCylinder(pcyl, cylheight, cylrad);
        }

        R x_0 = 2.405; // first zero of Bessel function J_0(x)
        R q0 = x_0/cylrad; // q0 = h of Cheng
        R ps = 11.8575134409; // propagation speed, sol to 4+x^2 = q0^2
        R fun_exact_u(double *P, int i, int dom) { // h^2 = k^2 + \gam^2, E = E_0 exp(-\gam z)
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0)  // -1/q0^2 ps dEz/dx
                return ps/q0 * x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1)
                return ps/q0 * y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return std::cyl_bessel_j(0, q0 * r)*exp(-ps*z);
        }
        R fun_exact_curlu(double *P, int i, int dom) { // curlF_theta = -dF/dz, e_theta = -x/r e_x + y/r e_y, J_0' = -J_1 
            R x = P[0], y = P[1], z = P[2];
            float r = sqrt(x*x + y*y);
            if (i == 0) // dy(Fz) - dz(Fy)
                return (-q0 + ps*ps/q0)*y/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else if (i == 1) // dz(Fx) - dx(Fz)
                return (-ps*ps/q0 + q0)*x/r*std::cyl_bessel_j(1, q0 * r)*exp(-ps*z);
            else
                return 0; // dx(Fy) - dy(Fx)
        }
        R fun_rhs(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return 0;
            else if (i == 1)
                return 0;
            else
                return 0;
        }
        R fun_exact_divu(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }
        R fun_exact_p(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }

    }
    namespace Data_Easy {
        R fun_0(double *P, int i) {
            return 0;
        }

        R k = 2.;
        R eps = 1.;
        R mu = 1.;

        R fun_levelSet(double *P, int i) {
            return (P[0] - 0.5) * (P[0] - 0.5) + (P[1] - 0.5) * (P[1] - 0.5) +
                (P[2] - 0.5) * (P[2] - 0.5) - 0.35 * 0.35;
        }

        R fun_exact_u(double *P, int i, int dom) { 
            R x = P[0], y = P[1], z = P[2];
            if (i == 0) 
                return x;
            else if (i == 1)
                return -2*y;
            else
                return z;
        }
        R fun_exact_curlu(double *P, int i, int dom) { 
            R x = P[0], y = P[1], z = P[2];
            if (i == 0) // dy(Fz) - dz(Fy)
                return 0;
            else if (i == 1) // dz(Fx) - dx(Fz)
                return 0;
            else
                return 0; // dx(Fy) - dy(Fx)
        }
        R fun_rhs(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            if (i == 0)
                return -k*k*x;
            else if (i == 1)
                return +2*k*k*y;
            else
                return -k*k*z;
        }
        R fun_exact_divu(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }
        R fun_exact_p(double *P, int i, int dom) {
            R x = P[0], y = P[1], z = P[2];
            return 0;
        }

    }

    // using namespace Data_Cylinder;
    using namespace Data_Easy;
    int main(int argc, char **argv) {
        typedef TestFunction<Mesh3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;
        const double cpubegin = CPUtime();

        //MPIcf cfMPI(argc, argv);

        const int d = 3;

        int nx = 7;
        int ny = 7;
        int nz = 7;

        std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp;

        int iters = 2;
        for (int i = 0; i < iters; ++i) {
            R sh = M_PI*3e-2;
            // Mesh3 Kh(nx, ny, nz, -0.2-sh, -0.2-sh, 0.0-sh, 0.4+sh, 0.4+sh, 2.01*cylheight+sh);
            Mesh3 Kh(nx, ny, nz, 0, 0, 0, 1, 1, 1+sh*1e-10);

            const R hi = 1. / (nx - 1); // 1./(nx-1)

            Space Uh_(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
            Space Vh_(Kh, DataFE<Mesh>::RT0);
            Space Wh_(Kh, DataFE<Mesh>::P0);

            Lagrange3 VelocitySpace(2);
            Space Vel_h(Kh, VelocitySpace);

            Space Lh(Kh, DataFE<Mesh>::P1);
            Fun_h levelSet(Lh, fun_levelSet);
            InterfaceLevelSet<Mesh> interface(Kh, levelSet);
            Normal n;

            // [Remove exterior]
            ActiveMesh<Mesh> Khi(Kh);
            Khi.truncate(interface, 1);
            Khi.info();

            CutSpace Velh(Khi, Vel_h);
            CutSpace Uh(Khi, Uh_);
            CutSpace Vh(Khi, Vh_);
            CutSpace Wh(Khi, Wh_);

            // Interpolate data
            Fun_h fh(Velh, fun_rhs);
            Fun_h u0(Velh, fun_exact_u);
            Fun_h w0(Velh, fun_exact_curlu);

            // Init system matrix & assembly
            CutFEM<Mesh> maxwell3D(Uh);
            maxwell3D.add(Vh);
            maxwell3D.add(Wh);

            /* Syntax:
            FunTest (fem space, #components, place in space)
            */

            FunTest w(Uh, 3, 0), tau(Uh, 3, 0);
            FunTest u(Vh, 3, 0), v(Vh, 3, 0), p(Wh, 1, 0), q(Wh, 1, 0);
            R mu = 1.;

            // [Bulk]
            // Eq 1
            maxwell3D.addBilinear( // w = curl u
                +innerProduct(1. / mu * w, tau)
                -innerProduct(u, curl(tau))
            , Khi);
            // maxwell3D.addLinear(
            //     -innerProduct(cross(n, u0), tau)
            // , interface);
            R pp = 1e2;
            maxwell3D.addBilinear(
                -innerProduct(u, cross(n,tau))
                +innerProduct(cross(n,w), v)
                +innerProduct(cross(n,w), pp*1./hi * cross(n,tau))
            , interface);
            maxwell3D.addLinear(
                +innerProduct(cross(n,w0), v)
                +innerProduct(cross(n,w0), pp*1./hi * cross(n,tau))
            , interface);

            // Eq 2
            maxwell3D.addBilinear( // mu Delta u + grad p
                +innerProduct(curl(w), v)
                -innerProduct(k * k * eps * u, v)
                +innerProduct(p, div(v))
            , Khi);

            maxwell3D.addLinear(
                +innerProduct(fh.exprList(), v)
            , Khi);

            // Eq 3
            maxwell3D.addBilinear(
                -innerProduct(div(u), q)
            , Khi);

            // [Stabilization]
            // order 1 with mumps: 1e-1, 2e0, 1e-1, 15
            double tau_w = 1e0;       // smaller tau_w seems to give larger condition number
            double tau_m = 1e0;
            double tau_a = 1e0;
            double tau_b = 1e0;

            maxwell3D.addPatchStabilization(
                // W block
                +innerProduct(tau_w  * jump(w), jump(tau)) 
                +innerProduct(tau_m * hi * hi * jump(curl(w)), jump(curl(tau)))
                // M blocks
                +innerProduct(tau_m * jump(curl(w)), jump(v))                       // M block
                -innerProduct(tau_m * jump(u), jump(curl(tau)))                     // -M^T block
                // B blocks
                +innerProduct(tau_b * jump(p), jump(div(v)))                     // -B^T block
                -innerProduct(tau_b * jump(div(u)), jump(q))                     // B_0 block
            , Khi);//                                  , macro);

            matlab::Export(maxwell3D.mat_[0], "mat" + std::to_string(i) + ".dat");
            maxwell3D.solve("umfpack");

            // EXTRACT SOLUTION

            int nb_vort_dof = Uh.get_nb_dof();
            int nb_flux_dof = Vh.get_nb_dof();
            Rn_ data_wh = maxwell3D.rhs_(SubArray(nb_vort_dof, 0));
            Rn_ data_uh = maxwell3D.rhs_(SubArray(
                nb_flux_dof, nb_vort_dof)); 
            Rn_ data_ph = maxwell3D.rhs_(SubArray(Wh.get_nb_dof(), nb_vort_dof + nb_flux_dof)); //

            Fun_h wh(Uh, data_wh);
            Fun_h uh(Vh, data_uh);
            Fun_h ph(Wh, data_ph);

            auto uh_0dx = dx(uh.expr(0));
            auto uh_1dy = dy(uh.expr(1));
            auto uh_2dz = dz(uh.expr(2));

            // auto curl_uh = curl(uh); // vector of 3 expressionfunfems
            // std::cout << curl_uh.size() << std::endl;

            // [Paraview]
            {
                // Fun_h solw(Uh, fun_exact_w);
                Fun_h solu(Velh, fun_exact_u);
                Fun_h soluErr(Vh, fun_exact_u);
                Fun_h solp(Wh, fun_exact_p);

                soluErr.v -= uh.v;
                soluErr.v.map(fabs);

                // Fun_h divSolh(Wh, fun_div);
                // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

                Paraview<Mesh> writer(Khi, "maxwell_" + std::to_string(i) + ".vtk");

                writer.add(wh, "vorticity", 0, 3);
                writer.add(uh, "velocity", 0, 3);
                writer.add(ph, "pressure", 0, 1);

                // writer.add(dx_uh0+dy_uh1, "divergence");
                // writer.add(femSol_0dx+femSol_1dy+fflambdah, "divergence");

                writer.add(solp, "pressureExact", 0, 1);
                writer.add(solu, "velocityExact", 0, 2);
                writer.add(soluErr, "velocityError", 0, 2);
            }

            double errU      = L2normCut(uh, fun_exact_u, 0, 3);
            // double errU      = L2normCut(curl_uh, fun_exact_u, Khi);
            double errP      = L2normCut(ph, fun_exact_p, 0, 1);
            double errDiv    = L2normCut(uh_0dx + uh_1dy + uh_2dz, Khi);
            double maxErrDiv = maxNormCut(uh_0dx + uh_1dy + uh_2dz, Khi);

            ul2.push_back(errU);
            pl2.push_back(errP);
            divl2.push_back(errDiv);
            divmax.push_back(maxErrDiv);
            h.push_back(hi);
            if (i == 0) {
                convu.push_back(0);
                convp.push_back(0);
            } else {
                convu.push_back(log(ul2[i] / ul2[i - 1]) / log(h[i] / h[i - 1]));
                convp.push_back(log(pl2[i] / pl2[i - 1]) / log(h[i] / h[i - 1]));
            }
            nx = 2 * nx - 1;
            ny = 2 * ny - 1;
            nz = 2 * nz - 1;
        }
        std::cout << "\n"
            << std::left << std::setw(10) << std::setfill(' ') << "h" << std::setw(15) << std::setfill(' ')
            << "err p" << std::setw(15) << std::setfill(' ') << "conv p" << std::setw(15) << std::setfill(' ')
            << "err u" << std::setw(15) << std::setfill(' ') << "conv u" << std::setw(15) << std::setfill(' ')
            << "err divu"
            // << std::setw(15) << std::setfill(' ') << "conv divu"
            // << std::setw(15) << std::setfill(' ') << "err_new divu"
            // << std::setw(15) << std::setfill(' ') << "convLoc divu"
            << std::setw(15) << std::setfill(' ')
            << "err maxdivu"
            // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
            << "\n"
            << std::endl;
        for (int i = 0; i < h.size(); ++i) {
            std::cout << std::left << std::setw(10) << std::setfill(' ') << h[i] << std::setw(15) << std::setfill(' ')
                    << pl2[i] << std::setw(15) << std::setfill(' ') << convp[i] << std::setw(15) << std::setfill(' ')
                    << ul2[i] << std::setw(15) << std::setfill(' ') << convu[i] << std::setw(15) << std::setfill(' ')
                    << divl2[i]
                    // << std::setw(15) << std::setfill(' ') << convdivPr[i]
                    // << std::setw(15) << std::setfill(' ') << divPrintLoc[i]
                    // << std::setw(15) << std::setfill(' ') << convdivPrLoc[i]
                    << std::setw(15) << std::setfill(' ')
                    << divmax[i]
                    // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
                    << std::endl;
        }
    }
#endif
