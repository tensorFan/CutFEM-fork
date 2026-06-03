#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
// #include "../util/cputime.h"
// #ifdef USE_MPI
// #include "cfmpi.hpp"
// #endif

#include "finiteElement.hpp"
#include "baseProblem.hpp"
#include "paraview.hpp"
#include "../num/matlab.hpp"

// #include "../num/gnuplot.hpp"


// #define PROBLEM_UNFITTED_STOKES3D


// #define PROBLEM_UNFITTED_PRESROB2_STOKES_VORTICITY_4FIELD // (2023 autumn)
// #define PROBLEM_UNFITTED_HANSBO_STOKES_VORTICITY_4FIELD // (2023 autumn)

// #define PROBLEM_UNFITTED_2026_CURL // (2026 autumn)
// #define PROBLEM_UNFITTED_2026_CURL_3D_ABC // (2026 autumn, 3D ABC Beltrami test)
// #define PROBLEM_2026_KIKUCHI_3D_PRESROB_FITTED // Section 5.2 pressure-robustness test
#define PROBLEM_2026_KIKUCHI_3D_PRESROB_UNFITTED // pressure-robustness test, unfitted version


// 3D
#ifdef PROBLEM_UNFITTED_STOKES3D

    namespace Erik_Data_UNFITTED_STOKES3D {

    R3 shift(0., 0., 0.);
    R fun_levelSet(const R3 P, int i) {
        return sqrt((P.x - shift.x) * (P.x - shift.x) + (P.y - shift.y) * (P.y - shift.y) +
                    (P.z - shift.z) * (P.z - shift.z)) -
            2. / 3;
    }
    R fun_rhs(const R3 P, int i) { return 0; }
    R fun_boundary(const R3 P, int i) { return (i == 0) ? 0.5 * P.z : 0; }

    R fun_kkk(const R3 P, int i) { return 0.5 * P.z; }
    } // namespace Erik_Data_UNFITTED_STOKES3D
    using namespace Erik_Data_UNFITTED_STOKES3D;

    int main(int argc, char **argv) {
        typedef TestFunction<3> FunTest;
        typedef FunFEM<Mesh3> Fun_h;
        typedef Mesh3 Mesh;
        typedef ActiveMeshT3 CutMesh;
        typedef FESpace3 Space;
        typedef CutFESpaceT3 CutSpace;

        const double cpubegin = CPUtime();
        MPIcf cfMPI(argc, argv);

        const int d = 3;
        int nx      = 10;
        int ny      = 10;
        int nz      = 10;
        Mesh3 Kh(nx, ny, nz, -1., -1., -1., 2., 2., 2.);
        const R hi           = 1. / (nx - 1); // 1./(nx-1)
        const R penaltyParam = 4e3;           // 4e3, 8e2

        Space Uh_(Kh, DataFE<Mesh>::Ned0); // Nedelec order 0 type 1
        Space Vh_(Kh, DataFE<Mesh>::RT0);
        Space Wh_(Kh, DataFE<Mesh>::P0);

        Fun_h fh0(Uh_, fun_kkk);

        std::cout << fh0.v << std::endl;

        Paraview<Mesh> writer(Kh, "stokes3D_" + to_string(0) + ".vtk");
        writer.add(fh0, "kkk", 0, 3);

        return 0;

        // FEM<Mesh3> stokes3D_({&Uh_, &Vh_, &Wh_}); std::getchar();

        Space Lh(Kh, DataFE<Mesh>::P1);
        Fun_h levelSet(Lh, fun_levelSet);
        InterfaceLevelSet<Mesh> interface(Kh, levelSet);

        // [Remove exterior]
        ActiveMesh<Mesh> Khi(Kh);
        Khi.truncate(interface, 1);

        CutSpace Uh(Khi, Uh_);
        CutSpace Vh(Khi, Vh_);
        CutSpace Wh(Khi, Wh_);

        // Interpolate data
        Fun_h fh(Vh, fun_rhs);
        Fun_h u0(Vh, fun_boundary);

        // Init system matrix & assembly
        CutFEM<Mesh> stokes3D(Uh);
        stokes3D.add(Vh);
        stokes3D.add(Wh);
        // CutFEM<Mesh> stokes3D(Vh); stokes3D.add(Wh);

        Normal n;
        /* Syntax:
        FunTest (fem space, #components, place in space)
        */
        FunTest w(Uh, 3, 0), tau(Uh, 3, 0);
        FunTest u(Vh, 3, 0), v(Vh, 3, 0), p(Wh, 1, 0), q(Wh, 1, 0);
        R mu = 1;

        // stokes3D.addBilinear( // w = curl u
        //   innerProduct(curl(u), v)
        //   , Khi
        // );

        // // [Bulk]
        stokes3D.addBilinear( // w = curl u
            innerProduct(1. / mu * w, tau) - innerProduct(u, curl(tau)), Khi);
        stokes3D.addBilinear( // mu Delta u + grad p
            innerProduct(curl(w), v) - innerProduct(p, div(v)), Khi);
        stokes3D.addLinear(+innerProduct(fh.expression(2), v), Khi);
        stokes3D.addBilinear(+innerProduct(div(u), q), Khi);
        // // [Dirichlet Velocity BC]
        // const MeshParameter &itf_h(Parameter::measureIntegral);
        // stokes3D.addBilinear( // int_Omg grad(p)*v = int_itf p v*t - int_Omg p div(v)
        //   + innerProduct(p, v*n)
        //   + innerProduct(1./hi*penaltyParam*u*n, v*n)
        //   // - innerProduct(u*t, tau)
        //   // + innerProduct(w, v*t)
        //   // + innerProduct(1./hi*penaltyParam*u, v)
        //   , interface
        // );
        stokes3D.addLinear(+innerProduct(cross(n, u0), tau) // [wtf why is + now correct..?]
                            + innerProduct(u0 * n, 1. / hi * penaltyParam * v * n)
                        // - innerProduct(u0*t,tau)
                        // + innerProduct(u0.expression(2), 1./hi*penaltyParam*v)
                        ,
                        interface);

        // // [Sets uniqueness of the pressure]
        // // R meanP = integral(Khi,exactp,0);
        // // stokes3D.addLagrangeMultiplier(
        // //   innerProduct(1, p), 0
        // //   , Khi
        // // );
        // // [Sets uniqueness of the pressure in another way such that divu = 0]
        // CutFEM<Mesh> lagr(Uh); lagr.add(Vh); lagr.add(Wh);
        // Rn zero_vec = lagr.rhs_;
        // lagr.addLinear(
        //   innerProduct(1, p)
        //   , Khi
        // );
        // Rn lag_row(lagr.rhs_); lagr.rhs_ = zero_vec;
        // lagr.addLinear(
        //   innerProduct(1, v*n)
        //   , interface
        // );
        // stokes3D.addLagrangeVecToRowAndCol(lag_row,lagr.rhs_,0);
        // // // [Stabilization]
        // // double wPenParam = 1e1; // 1e1
        // // double uPenParam = 1e1; // 1e-1 ~ 1/penParam (2e0 for (0,lamm,0))
        // // double pPenParam = 1e1; // 1e0 (2e0 for (0,lamm,0))
        // // FunTest grad2un = grad(grad(u)*n)*n;
        // // FunTest grad2wn = grad(grad(w)*n)*n;
        // // // // FunTest grad2pn = grad(grad(p)*n)*n;
        // // // // FunTest grad2divun = grad(grad(div(u))*n)*n;
        // // stokes3D.addFaceStabilization(
        // //   /* "Primal" stab: (lw,0,la) */
        // //   // innerProduct(uPenParam*pow(hi,1)*jump(w), jump(tau)) // [w in P1, continuous]
        // //   +innerProduct(wPenParam*pow(hi,3)*jump(grad(w)*n), jump(grad(tau)*n))
        // //   +innerProduct(uPenParam*pow(hi,5)*jump(grad2wn), jump(grad2wn))
        // //   +innerProduct(uPenParam*pow(hi,1)*jump(u), jump(v)) // [maybe should be 2k-1 if can scale pressure also]
        // //   +innerProduct(uPenParam*pow(hi,3)*jump(grad(u)*n), jump(grad(v)*n))
        // //   +innerProduct(uPenParam*pow(hi,5)*jump(grad2un), jump(grad2un))
        // //   -innerProduct(pPenParam*pow(hi,1)*jump(p), jump(div(v)))
        // //   +innerProduct(pPenParam*pow(hi,1)*jump(div(u)), jump(q))
        // //   -innerProduct(pPenParam*pow(hi,3)*jump(grad(p)), jump(grad(div(v))))
        // //   +innerProduct(pPenParam*pow(hi,3)*jump(grad(div(u))) , jump(grad(q)))

        // //   /* Mixed stab: (0,lm,0) */
        // //   // innerProduct(uPenParam*pow(hi,1)*jump(w), jump(tau)) // [w in P1, continuous]
        // //   // +innerProduct(wPenParam*pow(hi,3)*jump(grad(w)*n), jump(grad(tau)*n))
        // //   // +innerProduct(wPenParam*pow(hi,5)*jump(grad2wn), jump(grad2wn))
        // //   // +innerProduct(uPenParam*pow(hi,1)*jump(curl(w)), jump(v))
        // //   // -innerProduct(uPenParam*pow(hi,1)*jump(u), jump(curl(tau)))
        // //   // +innerProduct(uPenParam*pow(hi,3)*jump(grad(curl(w))), jump(grad(v)))
        // //   // -innerProduct(uPenParam*pow(hi,3)*jump(grad(u)), jump(grad(curl(tau))))
        // //   // -innerProduct(pPenParam*pow(hi,1)*jump(p), jump(div(v)))
        // //   // +innerProduct(pPenParam*pow(hi,1)*jump(div(u)), jump(q))
        // //   // -innerProduct(pPenParam*pow(hi,3)*jump(grad(p)), jump(grad(div(v))))
        // //   // +innerProduct(pPenParam*pow(hi,3)*jump(grad(div(u))), jump(grad(q)))

        // //   , Khi
        // //   , macro
        // // );

        // stokes3D.solve();

        // EXTRACT SOLUTION
        int nb_vort_dof = Uh.get_nb_dof();
        int nb_flux_dof = Vh.get_nb_dof();
        Rn_ data_wh     = stokes3D.rhs_(SubArray(nb_vort_dof, 0));
        Rn_ data_uh     = stokes3D.rhs_(SubArray(
            nb_flux_dof, nb_vort_dof)); // Rn_ data_uh = stokes.rhs_(SubArray(nb_vort_dof+nb_flux_dof,nb_vort_dof));
        Rn_ data_ph     = stokes3D.rhs_(SubArray(
            Wh.get_nb_dof(),
            nb_vort_dof +
                nb_flux_dof)); // Rn_ data_ph = stokes.rhs_(SubArray(stokes_.get_nb_dof(),nb_vort_dof+nb_flux_dof));
        Fun_h wh(Uh, data_wh);
        Fun_h uh(Vh, data_uh);
        Fun_h ph(Wh, data_ph);

        //   // [Post process pressure]
        //   R meanP = integral(Khi,exactp,0);
        //   ExpressionFunFEM<Mesh> fem_p(ph,0,op_id);
        //   R meanPfem = integral(Khi,fem_p,0);
        //   // std::cout << meanP << std::endl;
        //   CutFEM<Mesh2> post(Wh);
        //   post.addLinear(
        //     innerProduct(1,q)
        //     , Khi
        //   );
        //   R area = post.rhs_.sum();
        //   ph.v -= meanPfem/area;
        //   ph.v += meanP/area;

        //   ExpressionFunFEM<Mesh> dx_uh0(uh, 0, op_dx);
        //   ExpressionFunFEM<Mesh> dy_uh1(uh, 1, op_dy);

        // // [Paraview]
        // {
        //   // Fun_h solw(Uh, fun_exact_w);
        //   Fun_h solu(Vh, fun_exact_u); Fun_h soluErr(Vh, fun_exact_u);
        //   Fun_h solp(Wh, fun_exact_p);
        //   soluErr.v -= uh.v;
        //   soluErr.v.map(fabs);
        //   // Fun_h divSolh(Wh, fun_div);
        //   // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

        //   Paraview<Mesh> writer(Khi, "stokes_"+to_string(i)+".vtk");
        //   writer.add(wh, "vorticity" , 0, 1);
        //   writer.add(uh, "velocity" , 0, 2);
        //   writer.add(ph, "pressure" , 0, 1);
        //   writer.add(dx_uh0+dy_uh1, "divergence");
        //   // writer.add(femSol_0dx+femSol_1dy+fflambdah, "divergence");
        //   writer.add(solp, "pressureExact" , 0, 1);
        //   writer.add(solu, "velocityExact" , 0, 2);
        //   writer.add(soluErr, "velocityError" , 0, 2);
        //   // writer.add(solh, "velocityError" , 0, 2);

        //   // writer.add(fabs(femDiv, "divergenceError");
        // }
    }
#endif

// problem 3 with 4fields
#ifdef PROBLEM_UNFITTED_PRESROB2_STOKES_VORTICITY_4FIELD

  namespace Erik_Data_UNFITTED_STOKES_VORTICITY {

    R Ra = 1e2;

    R fun_levelSet(const R2 P, const int i) {
      return 1-P.y;
    }

    // [Example 1 from Neilan pressure robust paper]
    R fun_div(const R2 P, int i, int dom) {
      R x = P.x;
      R y = P.y;
      return 0;
    }
    R fun_rhs(const R2 P, int i, int dom) {
      R x = P.x;
      R y = P.y;
      if(i==0) return      0;
      else return Ra*(1-y+3*y*y);
    }
    R fun_exact_u(const R2 P, int i, int dom) {
      R x = P.x;
      R y = P.y;
      if(i==0)    return  0;
      else        return  0;
    }
    R fun_exact_p(const R2 P, int i, int dom ) {
      R x = P.x;
      R y = P.y;
      return Ra*(y*y*y-y*y/2+y-7./12);
    }
  }
  using namespace Erik_Data_UNFITTED_STOKES_VORTICITY;

  int main(int argc, char** argv ) {
    typedef TestFunction<Mesh2> FunTest;
    typedef FunFEM<Mesh2> Fun_h;
    typedef Mesh2 Mesh;
    typedef ActiveMeshT2 CutMesh;
    typedef FESpace2   Space;
    typedef CutFESpaceT2 CutSpace;

    const double cpubegin = CPUtime();
    // MPIcf cfMPI(argc,argv);

    int nx = 11;
    int ny = 11;
    // int d = 2;

    std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp, gradul2, convgrad;

    int iters = 3;
    for(int i=0;i<iters;++i) { // i<3

      std::cout << "\n ------------------------------------- " << std::endl;
      Mesh Kh(nx, ny, 0., 0., 1., 1.+1e-12);
      const R hi = 1./(nx-1); // 1./(nx-1)
      // const R penaltyParam = 8e2; // 4e3, 8e2

      Space Lh(Kh, DataFE<Mesh2>::P1);
      Fun_h levelSet(Lh, fun_levelSet);
      InterfaceLevelSet<Mesh> interface(Kh, levelSet);


      Lagrange2 FEvelocity(4);
      Space VELh_(Kh, FEvelocity);
      Space SCAh_(Kh, DataFE<Mesh>::P4);

      Space Uh_(Kh, DataFE<Mesh>::P1); // Nedelec order 0 type 1
      Space Vh_(Kh, DataFE<Mesh2>::RT0); 
      Space Wh_(Kh, DataFE<Mesh2>::P0);
      Space Whh_(Kh, DataFE<Mesh2>::P0); 

      // ACTIVE MESH
      ActiveMesh<Mesh> Khi(Kh);
      Khi.truncate(interface, -1);
      MacroElement<Mesh> macro(Khi, 1); // we use 0.25 for vorticity BC2

      CutSpace VELh(Khi, VELh_);
      CutSpace SCAh(Khi, SCAh_);

      CutSpace Uh(Khi, Uh_);
      CutSpace Vh(Khi, Vh_);
      CutSpace Wh(Khi, Wh_);

      Fun_h fh(VELh, fun_rhs); // interpolates fun_rhs to fh of type Fun_h
      Fun_h u0(VELh, fun_exact_u);
      Fun_h p0(SCAh, fun_exact_p); 
      
      // SURFACE MESH
      ActiveMesh<Mesh> Kh_itf(Kh);
      Kh_itf.createSurfaceMesh(interface);
      // MacroElementSurface<Mesh> macro_itf(Kh_itf, interface, 1); // 0.3
      CutSpace Wh_itf(Kh_itf, Whh_);

      // PROBLEM SETUP
      CutFEM<Mesh2> stokes(Vh); stokes.add(Wh); stokes.add(Uh); stokes.add(Wh_itf);

      Normal n;
      Tangent t;
      /* Syntax:
      FunTest (fem space, #components, place in space)
      */
      FunTest w(Uh,1,0), tau(Uh,1,0), u(Vh,2,0), v(Vh,2,0), p(Wh,1,0), q(Wh,1,0);
      FunTest p_itf(Wh_itf,1,0), q_itf(Wh_itf,1,0);

      R mu = 1;
      {
      // [Bulk]
      stokes.addBilinear( // w = curl u 
        + innerProduct(1./mu*w, tau)
        - innerProduct(u, rotgrad(tau))
        , Khi
      );
      stokes.addBilinear( // mu Delta u + grad p
        + innerProduct(rotgrad(w), v)
        - innerProduct(p, div(v))
        , Khi
      );
      stokes.addLinear(
        + innerProduct(fh.exprList(), v)
        , Khi
      );
      stokes.addBilinear(
        + innerProduct(div(u), q)
        , Khi
      );
      // [Stabilization]
      double wPenParam = 1e0; // 1e1
      double uPenParam = 1e0; // 1e-1 ~ 1/penParam (2e0 for (0,lamm,0))
      double pPenParam = 1e0; // 1e0 (2e0 for (0,lamm,0))
      FunTest grad2un = grad(grad(u)*n)*n;
      FunTest grad2wn = grad(grad(w)*n)*n;
      stokes.addFaceStabilization( 
        /* "Primal" stab: (lw,0,la) */
        // innerProduct(uPenParam*pow(hi,1)*jump(w), jump(tau)) // [w in P1, continuous]
        +innerProduct(wPenParam*pow(hi,3)*jump(grad(w)*n), jump(grad(tau)*n))
        +innerProduct(uPenParam*pow(hi,5)*jump(grad2wn), jump(grad2wn))
        +innerProduct(uPenParam*pow(hi,1)*jump(u), jump(v)) 
        +innerProduct(uPenParam*pow(hi,3)*jump(grad(u)*n), jump(grad(v)*n))
        +innerProduct(uPenParam*pow(hi,5)*jump(grad2un), jump(grad2un))

        -innerProduct(pPenParam*pow(hi,1)*jump(p), jump(div(v)))
        +innerProduct(pPenParam*pow(hi,1)*jump(div(u)), jump(q))
        -innerProduct(pPenParam*pow(hi,3)*jump(grad(p)), jump(grad(div(v))))
        +innerProduct(pPenParam*pow(hi,3)*jump(grad(div(u))) , jump(grad(q)))
        // +innerProduct(pPenParam*pow(hi,1)*jump(p), jump(q))
        // +innerProduct(pPenParam*pow(hi,3)*jump(grad(p)), jump(grad(q)))

        , Khi
        , macro
      );
      // [For paper:]
      stokes.addFaceStabilization( // [previously h^(2k+1) + macro]
        -innerProduct(pPenParam*pow(hi,0)*jump(p_itf), jump(q_itf))
        -innerProduct(pPenParam*pow(hi,2)*jump(grad(p_itf)*n), jump(grad(q_itf)*n)) 
      , Kh_itf
      // , macro_itf // somehow fails at last iteration when not using macro (due to umfpack maybe?)
      );
      stokes.addBilinear( 
        -innerProduct(pPenParam*pow(hi,1)*grad(p_itf)*n, grad(q_itf)*n) 
      , Kh_itf
      );
      // [Saras test (12/03/24):]
      // stokes.addFaceStabilization( // [previously h^(2k+1) + macro]
      //   -innerProduct(pPenParam*pow(hi,1)*jump(p_itf), jump(q_itf))
      //   -innerProduct(pPenParam*pow(hi,3)*jump(grad(p_itf)*n), jump(grad(q_itf)*n)) 
      // , Kh_itf
      // // , macro_itf // somehow fails at last iteration when not using macro (due to umfpack maybe?)
      // );
      // stokes.addBilinear( 
      //   -innerProduct(pPenParam*pow(hi,1)*grad(p_itf)*n, grad(q_itf)*n) 
      // , Kh_itf
      // );

      stokes.addBilinear(
        + innerProduct(p_itf, v*n)
        + innerProduct(u*n, q_itf)
        , interface
      );
      stokes.addBilinearIntersection(
        + innerProduct(p_itf, v*n)
        + innerProduct(u*n, q_itf)
        , Kh_itf, Khi, INTEGRAL_BOUNDARY
      );
      Fun_h u00(Vh, fun_exact_u);
      stokes.setDirichlet(u00, Khi.Th);
      // Sets uniqueness of the pressure
      R meanP = integral(Khi,p0,0);
      stokes.addLagrangeMultiplier(
        innerProduct(1, p), meanP
        , Khi
      );

      }
      // std::cout << integral(Khi,exactp,0) << std::endl;
      matlab::Export(stokes.mat_[0], "mat"+std::to_string(i)+"Cut.dat");
      stokes.solve("umfpack");

      // EXTRACT SOLUTION
      int nb_vort_dof = Uh.get_nb_dof();
      int nb_vel_dof = Vh.get_nb_dof();
      int nb_pres_dof = Wh.get_nb_dof();

      std::cout << "Lagrange multiplier value: " << std::endl;
      std::cout << stokes.rhs_(nb_pres_dof+nb_vel_dof+nb_vort_dof+Wh_itf.get_nb_dof())<< std::endl;

      // Rn_ data_wh = stokes.rhs_(SubArray(nb_vort_dof,0));
      // Rn_ data_uh = stokes.rhs_(SubArray(nb_vel_dof,nb_vort_dof));
      // Rn_ data_ph = stokes.rhs_(SubArray(nb_pres_dof,nb_vel_dof + nb_vort_dof));
      // Rn_ data_ph_itf = stokes.rhs_(SubArray(Wh_itf.get_nb_dof(),nb_pres_dof + nb_vel_dof + nb_vort_dof));
      Rn_ data_uh = stokes.rhs_(SubArray(nb_vel_dof,0));
      Rn_ data_ph = stokes.rhs_(SubArray(nb_pres_dof,nb_vel_dof));
      Rn_ data_wh = stokes.rhs_(SubArray(nb_vort_dof,nb_vel_dof + nb_pres_dof));
      Rn_ data_ph_itf = stokes.rhs_(SubArray(Wh_itf.get_nb_dof(),nb_pres_dof + nb_vel_dof + nb_vort_dof));
      Fun_h uh(Vh, data_uh);
      Fun_h ph(Wh, data_ph);
      Fun_h ph_itf(Wh_itf, data_ph_itf);
      // std::cout << data_ph_itf << std::endl;

      // [Post process pressure]
      // R meanP = integral(Khi,exactp,0);
      // ExpressionFunFEM<Mesh> fem_p(ph,0,op_id);
      // R meanPfem = integral(Khi,fem_p,0);
      // // std::cout << meanP << std::endl;
      // CutFEM<Mesh2> post(Wh);
      // post.addLinear(
      //   innerProduct(1,q)
      //   , Khi
      // ); 
      // R area = post.rhs_.sum();
      // ph.v -= meanPfem/area;
      // ph.v += meanP/area;

      auto uh_0dx = dx(uh.expr(0));
      auto uh_1dy = dy(uh.expr(1));   

      auto uh_0dy = dy(uh.expr(0));
      auto uh_1dx = dx(uh.expr(1));   

      // [Errors]
      {
        Fun_h soluErr(Vh, fun_exact_u);
        Fun_h soluh(Vh, fun_exact_u);
        soluErr.v -= uh.v;
        soluErr.v.map(fabs);
        // Fun_h divSolh(Wh, fun_div);
        // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

        Paraview<Mesh> writer(Khi, "stokes_"+std::to_string(i)+".vtk");
        writer.add(uh, "velocity" , 0, 2);
        writer.add(ph, "pressure" , 0, 1);
        writer.add(uh_0dx+uh_1dy, "divergence");
        writer.add(soluh, "velocityExact" , 0, 2);
        writer.add(soluErr, "velocityError" , 0, 2);
        // writer.add(solh, "velocityError" , 0, 2);

        // writer.add(ph_itf, "itf_pressure" , 0, 1);

        // writer.add(fabs(femDiv, "divergenceError");
      }

      R errU      = L2normCut(uh,fun_exact_u,0,2);
      R errGradU  = sqrt(integral(Khi,uh_0dx*uh_0dx+uh_0dy*uh_0dy+uh_1dx*uh_1dx+uh_1dy*uh_1dy,0));
      R errP      = L2normCut(ph,fun_exact_p,0,1);
      R errDiv    = L2normCut(uh_0dx+uh_1dy,fun_div,Khi);
      R maxErrDiv = maxNormCut(uh_0dx+uh_1dy,fun_div,Khi);
      // R errDiv    = L2normCut(femSol_0dx+femSol_1dy+fflambdah,fun_div,Khi);
      // R maxErrDiv = maxNormCut(femSol_0dx+femSol_1dy+fflambdah,fun_div,Khi);

      h.push_back(hi);
      ul2.push_back(errU);
      pl2.push_back(errP);
      divl2.push_back(errDiv);
      divmax.push_back(maxErrDiv);
      gradul2.push_back(errGradU);
      if(i==0) {convu.push_back(0); convp.push_back(0); convgrad.push_back(0);}
      else {
        convu.push_back( log(ul2[i]/ul2[i-1])/log(h[i]/h[i-1]));
        convp.push_back(log(pl2[i]/pl2[i-1])/log(h[i]/h[i-1]));
        convgrad.push_back(log(gradul2[i]/gradul2[i-1])/log(h[i]/h[i-1]));
      }

      nx = 2*nx-1;
      ny = 2*ny-1;
    }
    std::cout << "\n" << std::left
    << std::setw(10) << std::setfill(' ') << "h"
    << std::setw(15) << std::setfill(' ') << "err_p"
    << std::setw(15) << std::setfill(' ') << "conv p"
    << std::setw(15) << std::setfill(' ') << "err u"
    << std::setw(15) << std::setfill(' ') << "conv u"
    << std::setw(15) << std::setfill(' ') << "err divu"
    // << std::setw(15) << std::setfill(' ') << "conv divu"
    << std::setw(15) << std::setfill(' ') << "err maxdivu"
    // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
    << std::setw(15) << std::setfill(' ') << "err gradu"
    // << std::setw(15) << std::setfill(' ') << "conv gradu"
    << "\n" << std::endl;
    for(int i=0;i<h.size();++i) {
      std::cout << std::left
      << std::setw(10) << std::setfill(' ') << h[i]
      << std::setw(15) << std::setfill(' ') << pl2[i]
      << std::setw(15) << std::setfill(' ') << convp[i]
      << std::setw(15) << std::setfill(' ') << ul2[i]
      << std::setw(15) << std::setfill(' ') << convu[i]
      << std::setw(15) << std::setfill(' ') << divl2[i]
      // << std::setw(15) << std::setfill(' ') << convdivPr[i]
      << std::setw(15) << std::setfill(' ') << divmax[i]
      // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
      << std::setw(15) << std::setfill(' ') << gradul2[i]
    //   << std::setw(15) << std::setfill(' ') << convgrad[i] 
      << std::endl;
    }

  }
#endif

// problem 2 Hansbo circle with 4fields
#ifdef PROBLEM_UNFITTED_HANSBO_STOKES_VORTICITY_4FIELD

  namespace Erik_Data_CORIOLIS_STOKESRT {
    R shift = 0.5;
    // R interfaceRad = 0.25;//2./3; // not exactly 1/4 to avoid interface cutting exaclty a vertex
    R interfaceRad = 0.5-1e-12; // [<-- Olshanskii example sqrt(0.25)=0.5 ] 
    R fun_levelSet(const R2 P, const int i) {
      return sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift)) - interfaceRad;
    }

    // [Coriolis example]
    R fun_div(const R2 P, int i, int dom) {
      R x = P.x;
      R y = P.y;
      return 0;
    }
    R fun_rhs(const R2 P, int i, int dom) {
      // R mu=1;
      R x = P.x;
      R y = P.y;
      if(i==0) return      0;//100*(2-x)*(2-x);
      else if(i==1) return 0;
      else return 0;
    }
    R fun_exact_u(const R2 P, int i, int dom) {
      // R mu=1;
      R x = P.x;
      R y = P.y;
      if(i==0) return      1;
      else if(i==1) return 0;
      else return 0;
    }
  }

  using namespace Erik_Data_CORIOLIS_STOKESRT;
  int main(int argc, char** argv ) {
    typedef TestFunction<Mesh2> FunTest;
    typedef FunFEM<Mesh2> Fun_h;
    typedef Mesh2 Mesh;
    typedef ActiveMeshT2 CutMesh;
    typedef FESpace2   Space;
    typedef CutFESpaceT2 CutSpace;

    const double cpubegin = CPUtime();
    // MPIcf cfMPI(argc,argv);

    int nx = 11;
    int ny = 11;
    // int d = 2;

    std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp, gradul2, convgrad;

    int iters = 3;
    for(int i=0;i<iters;++i) { // i<3

      std::cout << "\n ------------------------------------- " << std::endl;
      Mesh Kh(nx, ny, 0., 0., 1., 1.);
      const R hi = 1./(nx-1); // 1./(nx-1)
      // const R penaltyParam = 8e2; // 4e3, 8e2

      Space Lh(Kh, DataFE<Mesh2>::P1);
      Fun_h levelSet(Lh, fun_levelSet);
      InterfaceLevelSet<Mesh> interface(Kh, levelSet);


      Lagrange2 FEvelocity(4);
      Space VELh_(Kh, FEvelocity);
      Space SCAh_(Kh, DataFE<Mesh>::P4);

      Space Uh_(Kh, DataFE<Mesh>::P1); // Nedelec order 0 type 1
      Space Vh_(Kh, DataFE<Mesh2>::RT0); 
      Space Wh_(Kh, DataFE<Mesh2>::P0);
      Space Whh_(Kh, DataFE<Mesh2>::P0);

      // ACTIVE MESH
      ActiveMesh<Mesh> Khi(Kh);
      Khi.truncate(interface, 1);
      MacroElement<Mesh> macro(Khi, 1); // we use 0.25 for vorticity BC2

      CutSpace VELh(Khi, VELh_);
      CutSpace SCAh(Khi, SCAh_);

      CutSpace Uh(Khi, Uh_);
      CutSpace Vh(Khi, Vh_);
      CutSpace Wh(Khi, Wh_);

      Fun_h fh(VELh, fun_rhs); // interpolates fun_rhs to fh of type Fun_h
      Fun_h u0(VELh, fun_exact_u);
    //   Fun_h p0(SCAh, fun_exact_p); 
      
      // SURFACE MESH
      ActiveMesh<Mesh> Kh_itf(Kh);
      Kh_itf.createSurfaceMesh(interface);
    //   MacroElementSurface<Mesh> macro_itf(Kh_itf, interface, 0.3); // 0.3
    //   MacroElementSurface<Mesh> macro_itf(interface, 0.8); // 0.3
      CutSpace Wh_itf(Kh_itf, Whh_);

      // PROBLEM SETUP
      CutFEM<Mesh2> stokes(Vh); stokes.add(Wh); stokes.add(Uh); stokes.add(Wh_itf);

      Normal n;
      Tangent t;
      /* Syntax:
      FunTest (fem space, #components, place in space)
      */
      FunTest w(Uh,1,0), tau(Uh,1,0), u(Vh,2,0), v(Vh,2,0), p(Wh,1,0), q(Wh,1,0);
      FunTest p_itf(Wh_itf,1,0), q_itf(Wh_itf,1,0);
      FunTest u1(Vh,1,0), u2(Vh,1,1), v1(Vh,1,0), v2(Vh,1,1);

      R mu = 0.01;
      R omega = 1e4;
      {
      // [Bulk]
      stokes.addBilinear( // coriolis
        - innerProduct(2*omega*u2,v1)
        + innerProduct(2*omega*u1,v2)
        , Khi
      );
      stokes.addBilinear( // w = curl u 
        + innerProduct(1./mu*w, tau)
        - innerProduct(u, rotgrad(tau))
        , Khi
      );
      stokes.addBilinear( // mu Delta u + grad p
        + innerProduct(rotgrad(w), v)
        - innerProduct(p, div(v))
        , Khi
      );
      stokes.addLinear(
        + innerProduct(fh.exprList(), v)
        , Khi
      );
      stokes.addBilinear(
        + innerProduct(div(u), q)
        , Khi
      );
      // [Stabilization]
      double wPenParam = 1e0; // 1e1
      double uPenParam = 1e0; // 1e-1 ~ 1/penParam (2e0 for (0,lamm,0))
      double pPenParam = 1e0; // 1e0 (2e0 for (0,lamm,0))
      FunTest grad2un = grad(grad(u)*n)*n;
      FunTest grad2wn = grad(grad(w)*n)*n;
      stokes.addFaceStabilization( 
        /* "Primal" stab: (lw,0,la) */
        // innerProduct(uPenParam*pow(hi,1)*jump(w), jump(tau)) // [w in P1, continuous]
        +innerProduct(wPenParam*pow(hi,3)*jump(grad(w)*n), jump(grad(tau)*n))
        +innerProduct(uPenParam*pow(hi,5)*jump(grad2wn), jump(grad2wn))
        +innerProduct(uPenParam*pow(hi,1)*jump(u), jump(v)) 
        +innerProduct(uPenParam*pow(hi,3)*jump(grad(u)*n), jump(grad(v)*n))
        +innerProduct(uPenParam*pow(hi,5)*jump(grad2un), jump(grad2un))

        -innerProduct(pPenParam*pow(hi,1)*jump(p), jump(div(v)))
        +innerProduct(pPenParam*pow(hi,1)*jump(div(u)), jump(q))
        -innerProduct(pPenParam*pow(hi,3)*jump(grad(p)), jump(grad(div(v))))
        +innerProduct(pPenParam*pow(hi,3)*jump(grad(div(u))) , jump(grad(q)))
        // +innerProduct(pPenParam*pow(hi,1)*jump(p), jump(q))
        // +innerProduct(pPenParam*pow(hi,3)*jump(grad(p)), jump(grad(q)))

        , Khi
        , macro
      );
      stokes.addFaceStabilization( // [previously h^(2k+1) + macro]
        -innerProduct(pPenParam*pow(hi,0)*jump(p_itf), jump(q_itf))
        -innerProduct(pPenParam*pow(hi,2)*jump(grad(p_itf)*n), jump(grad(q_itf)*n)) 
      , Kh_itf
    //   , macro_itf // somehow fails at last iteration when not using macro (due to umfpack maybe?)
      );
      // stokes.addBilinear( 
      //   -innerProduct(pPenParam*pow(hi,1)*grad(p_itf)*n, grad(q_itf)*n) 
      // , Kh_itf
      // );
    //    stokes.addBilinear( 
    //     -innerProduct(pPenParam*pow(hi,2)*grad(p_itf)*n, grad(q_itf)*n) 
    //   , interface
    //   );

      stokes.addBilinear(
        + innerProduct(p_itf, v*n)
        + innerProduct(u*n, q_itf)
        , interface
      );
      stokes.addLinear(
        + innerProduct(u0*t, tau)
        + innerProduct(u0*n, q_itf)
        , interface
      );      
      // Sets uniqueness of the pressure
      R meanP = 0;//integral(Khi,p0,0);
      stokes.addLagrangeMultiplier(
        innerProduct(1, p), meanP
        , Khi
      );

      }

      // std::cout << integral(Khi,exactp,0) << std::endl;
      matlab::Export(stokes.mat_[0], "mat"+std::to_string(i)+"Cut.dat");
      stokes.solve("umfpack");

      // EXTRACT SOLUTION
      int nb_vort_dof = Uh.get_nb_dof();
      int nb_vel_dof = Vh.get_nb_dof();
      int nb_pres_dof = Wh.get_nb_dof();

      std::cout << "Lagrange multiplier value: " << std::endl;
      std::cout << stokes.rhs_(nb_pres_dof+nb_vel_dof+nb_vort_dof+Wh_itf.get_nb_dof())<< std::endl;

      // Rn_ data_wh = stokes.rhs_(SubArray(nb_vort_dof,0));
      // Rn_ data_uh = stokes.rhs_(SubArray(nb_vel_dof,nb_vort_dof));
      // Rn_ data_ph = stokes.rhs_(SubArray(nb_pres_dof,nb_vel_dof + nb_vort_dof));
      // Rn_ data_ph_itf = stokes.rhs_(SubArray(Wh_itf.get_nb_dof(),nb_pres_dof + nb_vel_dof + nb_vort_dof));
      Rn_ data_uh = stokes.rhs_(SubArray(nb_vel_dof,0));
      Rn_ data_ph = stokes.rhs_(SubArray(nb_pres_dof,nb_vel_dof));
      Rn_ data_wh = stokes.rhs_(SubArray(nb_vort_dof,nb_vel_dof + nb_pres_dof));
      Rn_ data_ph_itf = stokes.rhs_(SubArray(Wh_itf.get_nb_dof(),nb_pres_dof + nb_vel_dof + nb_vort_dof));
      Fun_h uh(Vh, data_uh);
      Fun_h ph(Wh, data_ph);
      Fun_h ph_itf(Wh_itf, data_ph_itf);
      // std::cout << data_ph_itf << std::endl;

      // [Post process pressure]
      // R meanP = integral(Khi,exactp,0);
      // ExpressionFunFEM<Mesh> fem_p(ph,0,op_id);
      // R meanPfem = integral(Khi,fem_p,0);
      // // std::cout << meanP << std::endl;
      // CutFEM<Mesh2> post(Wh);
      // post.addLinear(
      //   innerProduct(1,q)
      //   , Khi
      // ); 
      // R area = post.rhs_.sum();
      // ph.v -= meanPfem/area;
      // ph.v += meanP/area;

      auto uh_0dx = dx(uh.expr(0));
      auto uh_1dy = dy(uh.expr(1));   

      auto uh_0dy = dy(uh.expr(0));
      auto uh_1dx = dx(uh.expr(1));   

      // [Errors]
      {
        Fun_h soluErr(Vh, fun_exact_u);
        Fun_h soluh(Vh, fun_exact_u);
        soluErr.v -= uh.v;
        soluErr.v.map(fabs);
        // Fun_h divSolh(Wh, fun_div);
        // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

        Paraview<Mesh> writer(Khi, "stokes_"+std::to_string(i)+".vtk");
        writer.add(uh, "velocity" , 0, 2);
        writer.add(ph, "pressure" , 0, 1);
        writer.add(uh_0dx+uh_1dy, "divergence");
        writer.add(soluh, "velocityExact" , 0, 2);
        writer.add(soluErr, "velocityError" , 0, 2);
        // writer.add(solh, "velocityError" , 0, 2);

        // writer.add(ph_itf, "itf_pressure" , 0, 1);

        // writer.add(fabs(femDiv, "divergenceError");
      }

    //   R errU      = L2normCut(uh,fun_exact_u,0,2);
      R errU      = maxNormCut(uh.expr(1),Khi);
      R errGradU  = sqrt(integral(Khi,uh_0dx*uh_0dx+uh_0dy*uh_0dy+uh_1dx*uh_1dx+uh_1dy*uh_1dy,0));
    //   R errP      = 0;//L2normCut(ph,fun_exact_p,0,1);
      R errDiv    = L2normCut(uh_0dx+uh_1dy,fun_div,Khi);
      R maxErrDiv = maxNormCut(uh_0dx+uh_1dy,fun_div,Khi);
      // R errDiv    = L2normCut(femSol_0dx+femSol_1dy+fflambdah,fun_div,Khi);
      // R maxErrDiv = maxNormCut(femSol_0dx+femSol_1dy+fflambdah,fun_div,Khi);

      h.push_back(hi);
      ul2.push_back(errU);
    //   pl2.push_back(errP);
      divl2.push_back(errDiv);
      divmax.push_back(maxErrDiv);
    //   gradul2.push_back(errGradU);
      if(i==0) {convu.push_back(0);}// convp.push_back(0); convgrad.push_back(0);}
      else {
        convu.push_back( log(ul2[i]/ul2[i-1])/log(h[i]/h[i-1]));
        // convp.push_back(log(pl2[i]/pl2[i-1])/log(h[i]/h[i-1]));
        // convgrad.push_back(log(gradul2[i]/gradul2[i-1])/log(h[i]/h[i-1]));
      }

      nx = 2*nx-1;
      ny = 2*ny-1;
    }
    std::cout << "\n" << std::left
    << std::setw(10) << std::setfill(' ') << "h"
    // << std::setw(15) << std::setfill(' ') << "err_p"
    // << std::setw(15) << std::setfill(' ') << "conv p"
    << std::setw(15) << std::setfill(' ') << "err u"
    << std::setw(15) << std::setfill(' ') << "conv u"
    << std::setw(15) << std::setfill(' ') << "err divu"
    // << std::setw(15) << std::setfill(' ') << "conv divu"
    << std::setw(15) << std::setfill(' ') << "err maxdivu"
    // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
    // << std::setw(15) << std::setfill(' ') << "err gradu"
    // << std::setw(15) << std::setfill(' ') << "conv gradu"
    << "\n" << std::endl;
    for(int i=0;i<h.size();++i) {
      std::cout << std::left
      << std::setw(10) << std::setfill(' ') << h[i]
    //   << std::setw(15) << std::setfill(' ') << pl2[i]
    //   << std::setw(15) << std::setfill(' ') << convp[i]
      << std::setw(15) << std::setfill(' ') << ul2[i]
      << std::setw(15) << std::setfill(' ') << convu[i]
      << std::setw(15) << std::setfill(' ') << divl2[i]
      // << std::setw(15) << std::setfill(' ') << convdivPr[i]
      << std::setw(15) << std::setfill(' ') << divmax[i]
      // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
    //   << std::setw(15) << std::setfill(' ') << gradul2[i]
    //   << std::setw(15) << std::setfill(' ') << convgrad[i] 
      << std::endl;
    }

  }
#endif



// 2026 Curl formulation Stokes example for possible reviewer comment
#ifdef PROBLEM_UNFITTED_2026_CURL

  namespace Erik_Data_UNFITTED_STOKES_VORTICITY {

    R Ra = 1e2;

    R fun_levelSet(const R2 P, const int i) {
      return 1-P.y;
    }

    // [Example 1 from Neilan pressure robust paper]
    R fun_div(const R2 P, int i, int dom) {
      R x = P.x;
      R y = P.y;
      return 0;
    }
    R fun_rhs(const R2 P, int i, int dom) {
      R x = P.x;
      R y = P.y;
      if(i==0) return      0;
      else return Ra*(1-y+3*y*y);
    }
    R fun_exact_u(const R2 P, int i, int dom) {
      R x = P.x;
      R y = P.y;
      if(i==0)    return  0;
      else        return  0;
    }
    R fun_exact_p(const R2 P, int i, int dom ) {
      R x = P.x;
      R y = P.y;
      return Ra*(y*y*y-y*y/2+y-7./12);
    }
  }
  using namespace Erik_Data_UNFITTED_STOKES_VORTICITY;

  int main(int argc, char** argv ) {
    typedef TestFunction<Mesh2> FunTest;
    typedef FunFEM<Mesh2> Fun_h;
    typedef Mesh2 Mesh;
    typedef ActiveMeshT2 CutMesh;
    typedef FESpace2   Space;
    typedef CutFESpaceT2 CutSpace;

    const double cpubegin = CPUtime();
    MPIcf cfMPI(argc,argv);

    int nx = 11;
    int ny = 11;
    // int d = 2;

    std::vector<double> ul2, pl2, divmax, divl2, h, convu, convp, gradul2, convgrad;

    int iters = 3;
    for(int i=0;i<iters;++i) { // i<3

      std::cout << "\n ------------------------------------- " << std::endl;
      Mesh Kh(nx, ny, 0., 0., 1., 1.+1e-12);
      const R hi = 1./(nx-1); // 1./(nx-1)
      // const R penaltyParam = 8e2; // 4e3, 8e2

      Space Lh(Kh, DataFE<Mesh2>::P1);
      Fun_h levelSet(Lh, fun_levelSet);
      InterfaceLevelSet<Mesh> interface(Kh, levelSet);


      Lagrange2 FEvelocity(4);
      Space VELh_(Kh, FEvelocity);
      Space SCAh_(Kh, DataFE<Mesh>::P2);

      Space Uh_(Kh, DataFE<Mesh>::P1); // Nedelec order 0 type 1
      Space Vh_(Kh, DataFE<Mesh2>::RT0); 
      Space Wh_(Kh, DataFE<Mesh2>::P0);

      // ACTIVE MESH
      ActiveMesh<Mesh> Khi(Kh);
      Khi.truncate(interface, -1);
      MacroElement<Mesh> macro(Khi, 1); // we use 0.25 for vorticity BC2

      CutSpace VELh(Khi, VELh_);
      CutSpace SCAh(Khi, SCAh_);

      CutSpace Uh(Khi, Uh_);
      CutSpace Vh(Khi, Vh_);
      CutSpace Wh(Khi, Wh_);

      Fun_h fh(VELh, fun_rhs); // interpolates fun_rhs to fh of type Fun_h
      Fun_h u0(VELh, fun_exact_u);
      Fun_h p0(SCAh, fun_exact_p); 
      
      // PROBLEM SETUP
      CutFEM<Mesh2> stokes(Vh); stokes.add(Wh); stokes.add(Uh);

      Normal n;
      Tangent t;
      /* Syntax:
      FunTest (fem space, #components, place in space)
      */
      FunTest w(Uh,1,0), tau(Uh,1,0), u(Vh,2,0), v(Vh,2,0), p(Wh,1,0), q(Wh,1,0);

      R mu = 1;
      {
      // [Bulk]
      stokes.addBilinear( // w = curl u 
        + innerProduct(1./mu*w, tau)
        - innerProduct(u, rotgrad(tau))
        , Khi
      );
      stokes.addBilinear( // mu Delta u + grad p
        + innerProduct(rotgrad(w), v)
        - innerProduct(p, div(v))
        , Khi
      );
      stokes.addLinear(
        + innerProduct(fh.exprList(), v)
        , Khi
      );
      stokes.addBilinear(
        + innerProduct(div(u), q)
        , Khi
      );
      // [Stabilization]
      double wPenParam = 1e0; // 1e1
      double uPenParam = 1e0; // 1e-1 ~ 1/penParam (2e0 for (0,lamm,0))
      double pPenParam = 1e0; // 1e0 (2e0 for (0,lamm,0))
      FunTest grad2un = grad(grad(u)*n)*n;
      FunTest grad2wn = grad(grad(w)*n)*n;
      stokes.addFaceStabilization( 
        /* "Primal" stab: (lw,0,la) */
        // innerProduct(uPenParam*pow(hi,1)*jump(w), jump(tau)) // [w in P1, continuous]
        +innerProduct(wPenParam*pow(hi,3)*jump(grad(w)*n), jump(grad(tau)*n))
        // +innerProduct(uPenParam*pow(hi,5)*jump(grad2wn), jump(grad2wn))
        +innerProduct(uPenParam*pow(hi,1)*jump(u), jump(v)) 
        +innerProduct(uPenParam*pow(hi,3)*jump(grad(u)*n), jump(grad(v)*n))
        // +innerProduct(uPenParam*pow(hi,5)*jump(grad2un), jump(grad2un))

        -innerProduct(pPenParam*pow(hi,1)*jump(p), jump(div(v)))
        +innerProduct(pPenParam*pow(hi,1)*jump(div(u)), jump(q))
        // -innerProduct(pPenParam*pow(hi,3)*jump(grad(p)), jump(grad(div(v))))
        // +innerProduct(pPenParam*pow(hi,3)*jump(grad(div(u))) , jump(grad(q)))
        // +innerProduct(pPenParam*pow(hi,1)*jump(p), jump(q))
        // +innerProduct(pPenParam*pow(hi,3)*jump(grad(p)), jump(grad(q)))

        , Khi
        , macro
      );

      // [BC]
      double penParam = 1e2;
      Fun_h u0(Vh, fun_exact_u);
      stokes.addBilinear(
        + innerProduct(p, v*n)
        + innerProduct(u*n, penParam*pow(hi,-1) * v*n)
        , interface
      );
      stokes.addLinear(
        + innerProduct(u0*n, penParam*pow(hi,-1) * v*n)
        , interface
      );
      stokes.addBilinear(
        + innerProduct(p, v*n)
        + innerProduct(u*n, penParam*pow(hi,-1) * v*n)
        , Khi, INTEGRAL_BOUNDARY
      );
      stokes.addLinear(
        + innerProduct(u0*n, penParam*pow(hi,-1) * v*n)
        , Khi, INTEGRAL_BOUNDARY
      );
      // Fun_h u00(Vh, fun_exact_u);
      // stokes.setDirichlet(u00, Khi.Th);
      // Sets uniqueness of the pressure
      R meanP = integral(Khi,p0,0); //returning segfault...
      stokes.addLagrangeMultiplier(
        innerProduct(1, p), meanP
        , Khi
      );

      }
      // std::cout << integral(Khi,exactp,0) << std::endl;
      // matlab::Export(stokes.mat_[0], "mat"+std::to_string(i)+"Cut.dat");
      stokes.solve("umfpack");

      // EXTRACT SOLUTION
      int nb_vort_dof = Uh.get_nb_dof();
      int nb_vel_dof = Vh.get_nb_dof();
      int nb_pres_dof = Wh.get_nb_dof();

      std::cout << "Lagrange multiplier value: " << std::endl;
      std::cout << stokes.rhs_(nb_pres_dof+nb_vel_dof+nb_vort_dof)<< std::endl;

      // Rn_ data_wh = stokes.rhs_(SubArray(nb_vort_dof,0));
      // Rn_ data_uh = stokes.rhs_(SubArray(nb_vel_dof,nb_vort_dof));
      // Rn_ data_ph = stokes.rhs_(SubArray(nb_pres_dof,nb_vel_dof + nb_vort_dof));
      Rn_ data_uh = stokes.rhs_(SubArray(nb_vel_dof,0));
      Rn_ data_ph = stokes.rhs_(SubArray(nb_pres_dof,nb_vel_dof));
      Rn_ data_wh = stokes.rhs_(SubArray(nb_vort_dof,nb_vel_dof + nb_pres_dof));
      Fun_h uh(Vh, data_uh);
      Fun_h ph(Wh, data_ph);
      // std::cout << data_ph_itf << std::endl;

      // [Post process pressure]
      // R meanP = integral(Khi,exactp,0);
      // ExpressionFunFEM<Mesh> fem_p(ph,0,op_id);
      // R meanPfem = integral(Khi,fem_p,0);
      // // std::cout << meanP << std::endl;
      // CutFEM<Mesh2> post(Wh);
      // post.addLinear(
      //   innerProduct(1,q)
      //   , Khi
      // ); 
      // R area = post.rhs_.sum();
      // ph.v -= meanPfem/area;
      // ph.v += meanP/area;

      auto uh_0dx = dx(uh.expr(0));
      auto uh_1dy = dy(uh.expr(1));   

      auto uh_0dy = dy(uh.expr(0));
      auto uh_1dx = dx(uh.expr(1));   

      // [Errors]
      {
        Fun_h soluErr(Vh, fun_exact_u);
        Fun_h soluh(Vh, fun_exact_u);
        soluErr.v -= uh.v;
        soluErr.v.map(fabs);
        // Fun_h divSolh(Wh, fun_div);
        // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

        Paraview<Mesh> writer(Khi, "stokes_"+std::to_string(i)+".vtk");
        writer.add(uh, "velocity" , 0, 2);
        writer.add(ph, "pressure" , 0, 1);
        writer.add(uh_0dx+uh_1dy, "divergence");
        writer.add(soluh, "velocityExact" , 0, 2);
        writer.add(soluErr, "velocityError" , 0, 2);
        // writer.add(solh, "velocityError" , 0, 2);

        // writer.add(ph_itf, "itf_pressure" , 0, 1);

        // writer.add(fabs(femDiv, "divergenceError");
      }

      R errU      = L2normCut(uh,fun_exact_u,0,2);
      R errGradU  = sqrt(integral(Khi,uh_0dx*uh_0dx+uh_0dy*uh_0dy+uh_1dx*uh_1dx+uh_1dy*uh_1dy,0));
      R errP      = L2normCut(ph,fun_exact_p,0,1);
      R errDiv    = L2normCut(uh_0dx+uh_1dy,fun_div,Khi);
      R maxErrDiv = maxNormCut(uh_0dx+uh_1dy,fun_div,Khi);
      // R errDiv    = L2normCut(femSol_0dx+femSol_1dy+fflambdah,fun_div,Khi);
      // R maxErrDiv = maxNormCut(femSol_0dx+femSol_1dy+fflambdah,fun_div,Khi);

      h.push_back(hi);
      ul2.push_back(errU);
      pl2.push_back(errP);
      divl2.push_back(errDiv);
      divmax.push_back(maxErrDiv);
      gradul2.push_back(errGradU);
      if(i==0) {convu.push_back(0); convp.push_back(0); convgrad.push_back(0);}
      else {
        convu.push_back( log(ul2[i]/ul2[i-1])/log(h[i]/h[i-1]));
        convp.push_back(log(pl2[i]/pl2[i-1])/log(h[i]/h[i-1]));
        convgrad.push_back(log(gradul2[i]/gradul2[i-1])/log(h[i]/h[i-1]));
      }

      nx = 2*nx-1;
      ny = 2*ny-1;
    }
    std::cout << "\n" << std::left
    << std::setw(10) << std::setfill(' ') << "h"
    << std::setw(15) << std::setfill(' ') << "err_p"
    << std::setw(15) << std::setfill(' ') << "conv p"
    << std::setw(15) << std::setfill(' ') << "err u"
    << std::setw(15) << std::setfill(' ') << "conv u"
    << std::setw(15) << std::setfill(' ') << "err divu"
    // << std::setw(15) << std::setfill(' ') << "conv divu"
    << std::setw(15) << std::setfill(' ') << "err maxdivu"
    // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
    << std::setw(15) << std::setfill(' ') << "err gradu"
    // << std::setw(15) << std::setfill(' ') << "conv gradu"
    << "\n" << std::endl;
    for(int i=0;i<h.size();++i) {
      std::cout << std::left
      << std::setw(10) << std::setfill(' ') << h[i]
      << std::setw(15) << std::setfill(' ') << pl2[i]
      << std::setw(15) << std::setfill(' ') << convp[i]
      << std::setw(15) << std::setfill(' ') << ul2[i]
      << std::setw(15) << std::setfill(' ') << convu[i]
      << std::setw(15) << std::setfill(' ') << divl2[i]
      // << std::setw(15) << std::setfill(' ') << convdivPr[i]
      << std::setw(15) << std::setfill(' ') << divmax[i]
      // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
      << std::setw(15) << std::setfill(' ') << gradul2[i]
    //   << std::setw(15) << std::setfill(' ') << convgrad[i] 
      << std::endl;
    }

  }
#endif

// 2026 Curl formulation Stokes example in 3D.
// Manufactured Stokes data based on the Arnold--Beltrami--Childress flow
//     u = (A sin z + C cos y, B sin x + A cos z, C sin y + B cos x),
// for which div u = 0 and curl u = u when A=B=C=1.
// The pressure p = sin(x) sin(y) sin(z) has mean zero on the centred ball.
#ifdef PROBLEM_UNFITTED_2026_CURL_3D_ABC

  namespace Erik_Data_ABC_BELTRAMI_STOKES_3D {

    const R mu     = 1.0;
    const R Aabc   = 1.0;
    const R Babc   = 1.0;
    const R Cabc   = 1.0;
    const R radius = 0.78;
    const R eps_ls = 1e-14;

    R fun_levelSet(const R3 P, const int i) {
      return P.x*P.x + P.y*P.y + P.z*P.z - radius*radius + eps_ls;
    }

    R fun_0(const R3 P, const int i, const int dom) {
      return 0.0;
    }

    R fun_exact_u(const R3 P, const int i, const int dom) {
      const R x = P.x;
      const R y = P.y;
      const R z = P.z;

      if (i == 0) return Aabc*std::sin(z) + Cabc*std::cos(y);
      if (i == 1) return Babc*std::sin(x) + Aabc*std::cos(z);
      return Cabc*std::sin(y) + Babc*std::cos(x);
    }

    // For A=B=C=1 this agrees with curl u.  We keep the formula as mu*curl u
    // since the first equation in this block is w = mu curl u.
    R fun_exact_w(const R3 P, const int i, const int dom) {
      return mu*fun_exact_u(P, i, dom);
    }

    R fun_exact_p(const R3 P, const int i, const int dom) {
      const R x = P.x;
      const R y = P.y;
      const R z = P.z;
      return std::sin(x)*std::sin(y)*std::sin(z);
    }

    R fun_grad_p(const R3 P, const int i, const int dom) {
      const R x = P.x;
      const R y = P.y;
      const R z = P.z;

      if (i == 0) return std::cos(x)*std::sin(y)*std::sin(z);
      if (i == 1) return std::sin(x)*std::cos(y)*std::sin(z);
      return std::sin(x)*std::sin(y)*std::cos(z);
    }

    // Since curl(curl u) = u for the ABC Beltrami field, the Stokes forcing is
    //     f = curl(w) + grad p = mu*u + grad p.
    R fun_rhs(const R3 P, const int i, const int dom) {
      return mu*fun_exact_u(P, i, dom) + fun_grad_p(P, i, dom);
    }

    R fun_div(const R3 P, const int i, const int dom) {
      return 0.0;
    }

  } 
  using namespace Erik_Data_ABC_BELTRAMI_STOKES_3D;

  int main(int argc, char** argv) {
    typedef TestFunction<Mesh3> FunTest;
    typedef FunFEM<Mesh3> Fun_h;
    typedef Mesh3 Mesh;
    typedef ActiveMeshT3 CutMesh;
    typedef FESpace3 Space;
    typedef CutFESpaceT3 CutSpace;

    const double cpubegin = CPUtime();
    MPIcf cfMPI(argc, argv);

    int nx = 7;
    int ny = 7;
    int nz = 7;

    std::vector<double> wl2, ul2, pl2, divl2, divmax, h, convw, convu, convp;

    const int iters = 3;
    for (int i = 0; i < iters; ++i) {
      std::cout << "\n ------------------------------------- " << std::endl;
      std::cout << " --- 3D ABC Beltrami Stokes curl test --- " << std::endl;

      Mesh Kh(nx, ny, nz, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0);
      const R hi = 2.0/(nx - 1);

      Space Lh(Kh, DataFE<Mesh>::P1);
      Fun_h levelSet(Lh, fun_levelSet);
      InterfaceLevelSet<Mesh> interface(Kh, levelSet);

      Space Uh_(Kh, DataFE<Mesh>::Ned0); // H(curl), vorticity w
      Space Vh_(Kh, DataFE<Mesh>::RT0);  // H(div), velocity u
      Space Wh_(Kh, DataFE<Mesh>::P0);   // L2, pressure p

      Lagrange3 FEvelocity(2);
      Space VELh_(Kh, FEvelocity);

      ActiveMesh<Mesh> Khi(Kh);
      Khi.truncate(interface, 1);
      Khi.info();

      MacroElement<Mesh> macro(Khi, 0.25);

      CutSpace Uh(Khi, Uh_);
      CutSpace Vh(Khi, Vh_);
      CutSpace Wh(Khi, Wh_);
      CutSpace VELh(Khi, VELh_);

      Fun_h fh(VELh, fun_rhs);
      Fun_h u0(VELh, fun_exact_u);
      Fun_h w0(VELh, fun_exact_w);

      CutFEM<Mesh> stokes3D(Uh);
      stokes3D.add(Vh);
      stokes3D.add(Wh);

      Normal n;
      FunTest w(Uh, 3, 0), tau(Uh, 3, 0);
      FunTest u(Vh, 3, 0), v(Vh, 3, 0), p(Wh, 1, 0), q(Wh, 1, 0);

      std::cout << "Assembling..." << std::endl;
      // [Bulk]
      // First-order Stokes curl system:
      //     w = mu curl u,
      //     curl w + grad p = f,
      //     div u = 0.
      stokes3D.addBilinear(
        + innerProduct(1.0/mu*w, tau)
        - innerProduct(u, curl(tau))
        , Khi
      );
      stokes3D.addBilinear(
        + innerProduct(curl(w), v)
        - innerProduct(p, div(v))
        , Khi
      );
      stokes3D.addLinear(
        + innerProduct(fh.exprList(), v)
        , Khi
      );
      stokes3D.addBilinear(
        + innerProduct(div(u), q)
        , Khi
      );

      // [Unfitted boundary conditions]
      // Tangential velocity is inserted through the weak curl identity.
      // Normal velocity is imposed by a normal penalty in the H(div) block.
      const R penNormal = 1e2;
      stokes3D.addLinear(
        + innerProduct(cross(n, u0), tau)
        , interface
      );
      stokes3D.addBilinear(
        + innerProduct(p, v*n)
        + innerProduct(u*n, penNormal/hi * v*n)
        , interface
      );
      stokes3D.addLinear(
        + innerProduct(u0*n, penNormal/hi * v*n)
        , interface
      );

      // [Ghost penalties]
      // The terms mirror the 2D curl Stokes block and the Maxwell 3-field file,
      // with H(curl), H(div), and L2 pressure contributions.
      const R wPenParam = 1e-2;
      const R uPenParam = 1e-2;
      const R pPenParam = 1e-2;
      stokes3D.addPatchStabilization(
        + innerProduct(wPenParam * jump(w), jump(tau))
        // + innerProduct(wPenParam * jump(grad(w)*n), jump(grad(tau)*n))
        // + innerProduct(uPenParam * jump(u), jump(v))
        // + innerProduct(uPenParam * jump(grad(u)*n), jump(grad(v)*n))
        + innerProduct(uPenParam * jump(curl(w)), jump(v))
        - innerProduct(uPenParam * jump(u), jump(curl(tau)))
        - innerProduct(pPenParam * jump(p), jump(div(v)))
        + innerProduct(pPenParam * jump(div(u)), jump(q))
        , Khi
        , macro
      );

      // p has zero mean on the centred ball for this manufactured example.
      stokes3D.addLagrangeMultiplier(
        innerProduct(1, p), 0.0
        , Khi
      );

      // std::cout << "Exporting..." << std::endl;
      // matlab::Export(stokes3D.mat_[0], "mat3D_abc_" + std::to_string(i) + "Cut.dat");
      std::cout << "Solving..." << std::endl;
      stokes3D.solve("mumps");

      const int nb_vort_dof = Uh.get_nb_dof();
      const int nb_vel_dof  = Vh.get_nb_dof();
      const int nb_pres_dof = Wh.get_nb_dof();

      std::cout << "Lagrange multiplier value: " << std::endl;
      std::cout << stokes3D.rhs_(nb_vort_dof + nb_vel_dof + nb_pres_dof) << std::endl;

      Rn_ data_wh = stokes3D.rhs_(SubArray(nb_vort_dof, 0));
      Rn_ data_uh = stokes3D.rhs_(SubArray(nb_vel_dof, nb_vort_dof));
      Rn_ data_ph = stokes3D.rhs_(SubArray(nb_pres_dof, nb_vort_dof + nb_vel_dof));

      Fun_h wh(Uh, data_wh);
      Fun_h uh(Vh, data_uh);
      Fun_h ph(Wh, data_ph);

      auto uh_0dx = dx(uh.expr(0));
      auto uh_1dy = dy(uh.expr(1));
      auto uh_2dz = dz(uh.expr(2));

      {
        Fun_h solw(VELh, fun_exact_w);
        Fun_h solu(VELh, fun_exact_u);
        Fun_h solp(Wh, fun_exact_p);

        Fun_h solwErr(Uh, fun_exact_w);
        Fun_h soluErr(Vh, fun_exact_u);
        Fun_h solpErr(Wh, fun_exact_p);
        solwErr.v -= wh.v;
        soluErr.v -= uh.v;
        solpErr.v -= ph.v;
        solwErr.v.map(fabs);
        soluErr.v.map(fabs);
        solpErr.v.map(fabs);

        Paraview<Mesh> writer(Khi, "stokes3D_abc_" + std::to_string(i) + ".vtk");
        writer.add(wh, "vorticity", 0, 3);
        writer.add(uh, "velocity", 0, 3);
        writer.add(ph, "pressure", 0, 1);
        writer.add(uh_0dx + uh_1dy + uh_2dz, "divergence");
        writer.add(solw, "vorticityExact", 0, 3);
        writer.add(solu, "velocityExact", 0, 3);
        writer.add(solp, "pressureExact", 0, 1);
        writer.add(solwErr, "vorticityError", 0, 3);
        writer.add(soluErr, "velocityError", 0, 3);
        writer.add(solpErr, "pressureError", 0, 1);
      }

      const R errW      = L2normCut(wh, fun_exact_w, 0, 3);
      const R errU      = L2normCut(uh, fun_exact_u, 0, 3);
      const R errP      = L2normCut(ph, fun_exact_p, 0, 1);
      const R errDiv    = L2normCut(uh_0dx + uh_1dy + uh_2dz, Khi);
      const R maxErrDiv = maxNormCut(uh_0dx + uh_1dy + uh_2dz, Khi);

      h.push_back(hi);
      wl2.push_back(errW);
      ul2.push_back(errU);
      pl2.push_back(errP);
      divl2.push_back(errDiv);
      divmax.push_back(maxErrDiv);

      if (i == 0) {
        convw.push_back(0.0);
        convu.push_back(0.0);
        convp.push_back(0.0);
      } else {
        convw.push_back(std::log(wl2[i]/wl2[i-1])/std::log(h[i]/h[i-1]));
        convu.push_back(std::log(ul2[i]/ul2[i-1])/std::log(h[i]/h[i-1]));
        convp.push_back(std::log(pl2[i]/pl2[i-1])/std::log(h[i]/h[i-1]));
      }

      nx = 2*nx - 1;
      ny = 2*ny - 1;
      nz = 2*nz - 1;
    }

    std::cout << "\n" << std::left
      << std::setw(10) << std::setfill(' ') << "h"
      << std::setw(15) << std::setfill(' ') << "err p"
      << std::setw(15) << std::setfill(' ') << "conv p"
      << std::setw(15) << std::setfill(' ') << "err w"
      << std::setw(15) << std::setfill(' ') << "conv w"
      << std::setw(15) << std::setfill(' ') << "err u"
      << std::setw(15) << std::setfill(' ') << "conv u"
      << std::setw(15) << std::setfill(' ') << "err divu"
      << std::setw(15) << std::setfill(' ') << "err maxdivu"
      << "\n" << std::endl;

    for (int i = 0; i < h.size(); ++i) {
      std::cout << std::left
        << std::setw(10) << std::setfill(' ') << h[i]
        << std::setw(15) << std::setfill(' ') << pl2[i]
        << std::setw(15) << std::setfill(' ') << convp[i]
        << std::setw(15) << std::setfill(' ') << wl2[i]
        << std::setw(15) << std::setfill(' ') << convw[i]
        << std::setw(15) << std::setfill(' ') << ul2[i]
        << std::setw(15) << std::setfill(' ') << convu[i]
        << std::setw(15) << std::setfill(' ') << divl2[i]
        << std::setw(15) << std::setfill(' ') << divmax[i]
        << std::endl;
    }

    std::cout << "CPU time = " << CPUtime() - cpubegin << std::endl;
  }
#endif

// -----------------------------------------------------------------------------
// 2026 Kikuchi-style H(curl)-H1 Stokes test in 3D.
//
// This block implements the two-field curl formulation used in the paper
//   Find (u,p) in H(curl) x H^1_0mean such that
//       (curl u, curl v) + (grad p, v) = (f, v),
//       (u, grad q) = 0.
//
// The manufactured data are the 3D pressure-robustness test from Section 5.2:
//   u = ( 1/2 sin(2*pi*x) cos(2*pi*y) cos(2*pi*z),
//         1/2 cos(2*pi*x) sin(2*pi*y) cos(2*pi*z),
//        -    cos(2*pi*x) cos(2*pi*y) sin(2*pi*z) ),
//   p = lambda sin(2*pi*x) sin(2*pi*y) sin(2*pi*z), lambda = 1e5.
// Since div u = 0, curl(curl u) = -Delta u = 12*pi^2 u, so
//   f = curl(curl u) + grad p.
//
// IMPORTANT: the load is interpolated in the H(curl) space before addLinear:
//   Fun_h fh(Uh, fun_rhs);
//   stokes.addLinear(innerProduct(fh.exprList(), v), Khi);
// Do not replace this by a Lagrange/vector L2 interpolation if testing pressure
// robustness; the point is to feed the right-hand side through H(curl) DoFs.
// -----------------------------------------------------------------------------
#ifdef PROBLEM_2026_KIKUCHI_3D_PRESROB_FITTED

  namespace Erik_Data_KIKUCHI_3D_PRESROB {

    const R pi    = std::acos(-1.0);
    const R twopi = 2.0*pi;
    const R curlcurl_coeff = 3.0*twopi*twopi; // = 12*pi^2

    // Section 5.2 uses lambda = 10^5.  The value can be overridden by argv[1].
    R pressureLambda = 1e5;

    R fun_exact_u(const R3 P, const int i, const int dom) {
      const R sx = std::sin(twopi*P.x);
      const R cx = std::cos(twopi*P.x);
      const R sy = std::sin(twopi*P.y);
      const R cy = std::cos(twopi*P.y);
      const R sz = std::sin(twopi*P.z);
      const R cz = std::cos(twopi*P.z);

      if (i == 0) return 0.5*sx*cy*cz;
      if (i == 1) return 0.5*cx*sy*cz;
      return -cx*cy*sz;
    }

    R fun_exact_p(const R3 P, const int i, const int dom) {
      return pressureLambda
        * std::sin(twopi*P.x)
        * std::sin(twopi*P.y)
        * std::sin(twopi*P.z);
    }

    R fun_grad_p(const R3 P, const int i, const int dom) {
      const R sx = std::sin(twopi*P.x);
      const R cx = std::cos(twopi*P.x);
      const R sy = std::sin(twopi*P.y);
      const R cy = std::cos(twopi*P.y);
      const R sz = std::sin(twopi*P.z);
      const R cz = std::cos(twopi*P.z);

      if (i == 0) return pressureLambda*twopi*cx*sy*sz;
      if (i == 1) return pressureLambda*twopi*sx*cy*sz;
      return pressureLambda*twopi*sx*sy*cz;
    }

    R fun_curlcurl_u(const R3 P, const int i, const int dom) {
      return curlcurl_coeff*fun_exact_u(P, i, dom);
    }

    R fun_rhs(const R3 P, const int i, const int dom) {
      return fun_curlcurl_u(P, i, dom) + fun_grad_p(P, i, dom);
    }

    R fun_exact_curl_u(const R3 P, const int i, const int dom) {
      const R sx = std::sin(twopi*P.x);
      const R cx = std::cos(twopi*P.x);
      const R sy = std::sin(twopi*P.y);
      const R cy = std::cos(twopi*P.y);
      const R sz = std::sin(twopi*P.z);

      if (i == 0) return 1.5*twopi*cx*sy*sz;
      if (i == 1) return -1.5*twopi*sx*cy*sz;
      return 0.0;
    }

    R fun_exact_curl_u_0(const R3 P, const int i, const int dom) {
      return fun_exact_curl_u(P, 0, dom);
    }
    R fun_exact_curl_u_1(const R3 P, const int i, const int dom) {
      return fun_exact_curl_u(P, 1, dom);
    }
    R fun_exact_curl_u_2(const R3 P, const int i, const int dom) {
      return fun_exact_curl_u(P, 2, dom);
    }

    R fun_grad_p_0(const R3 P, const int i, const int dom) {
      return fun_grad_p(P, 0, dom);
    }
    R fun_grad_p_1(const R3 P, const int i, const int dom) {
      return fun_grad_p(P, 1, dom);
    }
    R fun_grad_p_2(const R3 P, const int i, const int dom) {
      return fun_grad_p(P, 2, dom);
    }

  }
  using namespace Erik_Data_KIKUCHI_3D_PRESROB;

  int main(int argc, char** argv) {
    typedef TestFunction<Mesh3> FunTest;
    typedef FunFEM<Mesh3> Fun_h;
    typedef Mesh3 Mesh;
    typedef ActiveMeshT3 CutMesh;
    typedef FESpace3 Space;
    typedef CutFESpaceT3 CutSpace;

    const double cpubegin = CPUtime();
    MPIcf cfMPI(argc, argv);

    if (argc > 1) {
      pressureLambda = std::atof(argv[1]);
    }

    int nx = 5;
    int ny = 5;
    int nz = 5;

    std::vector<double> ul2, curlul2, pl2, gradpl2, divl2, divmax;
    std::vector<double> h, convu, convcurlu, convp, convgradp;

    const int iters = 3;
    for (int i = 0; i < iters; ++i) {
      std::cout << "\n ------------------------------------- " << std::endl;
      std::cout << " --- 3D Kikuchi Hcurl-H1 pressure-robustness test --- " << std::endl;
      std::cout << " lambda = " << pressureLambda << std::endl;

      Mesh Kh(nx, ny, nz, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
      const R hi = 1.0/(nx - 1);

      Space Uh_(Kh, DataFE<Mesh>::Ned0); // H(curl) velocity, lowest-order Nedelec
      Space Ph_(Kh, DataFE<Mesh>::P1);   // H1 pressure

      // Full fitted cube.  We still use ActiveMesh/CutFESpace to match the rest
      // of stokesRT.cpp and to keep the same assembly API.
      ActiveMesh<Mesh> Khi(Kh);
      Khi.info();

      CutSpace Uh(Khi, Uh_);
      CutSpace Ph(Khi, Ph_);

      // Pressure-robust load handling: interpolate f into H(curl), not into a
      // vector Lagrange/L2 space, before putting it in the linear form.
      Fun_h fh(Uh, fun_rhs);

      CutFEM<Mesh> stokes(Uh);
      stokes.add(Ph);

      FunTest u(Uh, 3, 0), v(Uh, 3, 0);
      FunTest p(Ph, 1, 0), q(Ph, 1, 0);

      std::cout << "Assembling..." << std::endl;
      stokes.addBilinear(
        + innerProduct(curl(u), curl(v))
        + innerProduct(grad(p), v)
        , Khi
      );
      stokes.addLinear(
        + innerProduct(fh.exprList(), v)
        , Khi
      );
      stokes.addBilinear(
        + innerProduct(u, grad(q))
        , Khi
      );

      // The exact pressure has zero mean on (0,1)^3.
      stokes.addLagrangeMultiplier(
        innerProduct(1, p), 0.0
        , Khi
      );

      std::cout << "Solving..." << std::endl;
      stokes.solve("mumps");

      const int nb_vel_dof  = Uh.get_nb_dof();
      const int nb_pres_dof = Ph.get_nb_dof();

      std::cout << "Lagrange multiplier value: " << std::endl;
      std::cout << stokes.rhs_(nb_vel_dof + nb_pres_dof) << std::endl;

      Rn_ data_uh = stokes.rhs_(SubArray(nb_vel_dof, 0));
      Rn_ data_ph = stokes.rhs_(SubArray(nb_pres_dof, nb_vel_dof));

      Fun_h uh(Uh, data_uh);
      Fun_h ph(Ph, data_ph);

      auto uh_0dx = dx(uh.expr(0));
      auto uh_0dy = dy(uh.expr(0));
      auto uh_0dz = dz(uh.expr(0));
      auto uh_1dx = dx(uh.expr(1));
      auto uh_1dy = dy(uh.expr(1));
      auto uh_1dz = dz(uh.expr(1));
      auto uh_2dx = dx(uh.expr(2));
      auto uh_2dy = dy(uh.expr(2));
      auto uh_2dz = dz(uh.expr(2));

      auto curl_uh_0 = uh_2dy - uh_1dz;
      auto curl_uh_1 = uh_0dz - uh_2dx;
      auto curl_uh_2 = uh_1dx - uh_0dy;

      auto ph_dx = dx(ph.expr(0));
      auto ph_dy = dy(ph.expr(0));
      auto ph_dz = dz(ph.expr(0));

      {
        Fun_h solu(Uh, fun_exact_u);
        Fun_h solp(Ph, fun_exact_p);
        Fun_h soluErr(Uh, fun_exact_u);
        Fun_h solpErr(Ph, fun_exact_p);
        soluErr.v -= uh.v;
        solpErr.v -= ph.v;
        soluErr.v.map(fabs);
        solpErr.v.map(fabs);

        Paraview<Mesh> writer(Khi, "stokes3D_kikuchi_presrob_" + std::to_string(i) + ".vtk");
        writer.add(uh, "velocity", 0, 3);
        writer.add(ph, "pressure", 0, 1);
        writer.add(curl_uh_0, "curl_u_0");
        writer.add(curl_uh_1, "curl_u_1");
        writer.add(curl_uh_2, "curl_u_2");
        writer.add(uh_0dx + uh_1dy + uh_2dz, "divergence");
        writer.add(solu, "velocityExact", 0, 3);
        writer.add(solp, "pressureExact", 0, 1);
        writer.add(soluErr, "velocityError", 0, 3);
        writer.add(solpErr, "pressureError", 0, 1);
      }

      const R errU = L2normCut(uh, fun_exact_u, 0, 3);
      const R errCurlU = std::sqrt(
        std::pow(L2normCut(curl_uh_0, fun_exact_curl_u_0, Khi), 2)
        + std::pow(L2normCut(curl_uh_1, fun_exact_curl_u_1, Khi), 2)
        + std::pow(L2normCut(curl_uh_2, fun_exact_curl_u_2, Khi), 2)
      );
      const R errP = L2normCut(ph, fun_exact_p, 0, 1);
      const R errGradP = std::sqrt(
        std::pow(L2normCut(ph_dx, fun_grad_p_0, Khi), 2)
        + std::pow(L2normCut(ph_dy, fun_grad_p_1, Khi), 2)
        + std::pow(L2normCut(ph_dz, fun_grad_p_2, Khi), 2)
      );
      const R errDiv = L2normCut(uh_0dx + uh_1dy + uh_2dz, Khi);
      const R maxErrDiv = maxNormCut(uh_0dx + uh_1dy + uh_2dz, Khi);

      h.push_back(hi);
      ul2.push_back(errU);
      curlul2.push_back(errCurlU);
      pl2.push_back(errP);
      gradpl2.push_back(errGradP);
      divl2.push_back(errDiv);
      divmax.push_back(maxErrDiv);

      if (i == 0) {
        convu.push_back(0.0);
        convcurlu.push_back(0.0);
        convp.push_back(0.0);
        convgradp.push_back(0.0);
      } else {
        convu.push_back(std::log(ul2[i]/ul2[i-1])/std::log(h[i]/h[i-1]));
        convcurlu.push_back(std::log(curlul2[i]/curlul2[i-1])/std::log(h[i]/h[i-1]));
        convp.push_back(std::log(pl2[i]/pl2[i-1])/std::log(h[i]/h[i-1]));
        convgradp.push_back(std::log(gradpl2[i]/gradpl2[i-1])/std::log(h[i]/h[i-1]));
      }

      nx = 2*nx - 1;
      ny = 2*ny - 1;
      nz = 2*nz - 1;
    }

    std::cout << "\n" << std::left
      << std::setw(10) << std::setfill(' ') << "h"
      << std::setw(15) << std::setfill(' ') << "err u"
      << std::setw(15) << std::setfill(' ') << "conv u"
      << std::setw(15) << std::setfill(' ') << "err curlu"
      << std::setw(15) << std::setfill(' ') << "conv curlu"
      << std::setw(15) << std::setfill(' ') << "err p"
      << std::setw(15) << std::setfill(' ') << "conv p"
      << std::setw(15) << std::setfill(' ') << "err gradp"
      << std::setw(15) << std::setfill(' ') << "conv gradp"
      << std::setw(15) << std::setfill(' ') << "err divu"
      << std::setw(15) << std::setfill(' ') << "err maxdivu"
      << "\n" << std::endl;

    for (int i = 0; i < h.size(); ++i) {
      std::cout << std::left
        << std::setw(10) << std::setfill(' ') << h[i]
        << std::setw(15) << std::setfill(' ') << ul2[i]
        << std::setw(15) << std::setfill(' ') << convu[i]
        << std::setw(15) << std::setfill(' ') << curlul2[i]
        << std::setw(15) << std::setfill(' ') << convcurlu[i]
        << std::setw(15) << std::setfill(' ') << pl2[i]
        << std::setw(15) << std::setfill(' ') << convp[i]
        << std::setw(15) << std::setfill(' ') << gradpl2[i]
        << std::setw(15) << std::setfill(' ') << convgradp[i]
        << std::setw(15) << std::setfill(' ') << divl2[i]
        << std::setw(15) << std::setfill(' ') << divmax[i]
        << std::endl;
    }

    std::cout << "CPU time = " << CPUtime() - cpubegin << std::endl;
  }
#endif



// -----------------------------------------------------------------------------
// 2026 Kikuchi-style H(curl)-H1 Stokes test on a cut filled sphere.
//
// Purpose: pressure robustness in the genuinely cut case.  We use a pure
// gradient forcing
//      u_exact = 0,
//      p_exact = lambda * (|x|^2 - 3 R^2 / 5),
//      f       = grad p_exact = 2 lambda x.
// The exact pressure has zero mean on the exact filled ball of radius R, and the
// exact velocity satisfies the natural Kikuchi boundary conditions
//      u . n = 0,       curl(u) x n = 0
// on the sphere.  Hence any velocity growth with lambda is a direct sign that
// the pressure gradient is leaking into the velocity solve.
//
// IMPORTANT: f is interpolated into the H(curl) velocity space before addLinear.
// This is the key pressure-robust ingredient; do not replace fh by a vector
// Lagrange/L2 interpolation when running this test.
// -----------------------------------------------------------------------------
#ifdef PROBLEM_2026_KIKUCHI_3D_PRESROB_UNFITTED

  namespace Erik_Data_KIKUCHI_3D_PRESROB_SPHERE {

    const R radius = 2.0/3.0;
    const R radius2 = radius*radius;

    // Default value; override with argv[1], e.g. ./bin/stokesRT 1e7
    R pressureLambda = 1e5;

    // R fun_levelSet(const R3 P, const int i) {
    //   return std::sqrt(P.x*P.x + P.y*P.y + P.z*P.z) - radius;
    // }
    const R eps_ls = 1e-14;
    R fun_levelSet(const R3 P, const int i) {
      return P.x*P.x + P.y*P.y + P.z*P.z - radius*radius + eps_ls;
    }

    R fun_exact_u(const R3 P, const int i, const int dom) {
      return 0.0;
    }

    R fun_exact_p(const R3 P, const int i, const int dom) {
      const R r2 = P.x*P.x + P.y*P.y + P.z*P.z;
      // Average of r^2 over B_R in 3D is 3 R^2 / 5.
      return pressureLambda * (r2 - 3.0*radius2/5.0);
    }

    R fun_grad_p(const R3 P, const int i, const int dom) {
      if (i == 0) return 2.0*pressureLambda*P.x;
      if (i == 1) return 2.0*pressureLambda*P.y;
      return 2.0*pressureLambda*P.z;
    }

    R fun_rhs(const R3 P, const int i, const int dom) {
      return fun_grad_p(P, i, dom);
    }

    R fun_grad_p_0(const R3 P, const int i, const int dom) {
      return fun_grad_p(P, 0, dom);
    }
    R fun_grad_p_1(const R3 P, const int i, const int dom) {
      return fun_grad_p(P, 1, dom);
    }
    R fun_grad_p_2(const R3 P, const int i, const int dom) {
      return fun_grad_p(P, 2, dom);
    }

  }
  using namespace Erik_Data_KIKUCHI_3D_PRESROB_SPHERE;

  int main(int argc, char** argv) {
    typedef TestFunction<Mesh3> FunTest;
    typedef FunFEM<Mesh3> Fun_h;
    typedef Mesh3 Mesh;
    typedef ActiveMeshT3 CutMesh;
    typedef FESpace3 Space;
    typedef CutFESpaceT3 CutSpace;

    const double cpubegin = CPUtime();
    MPIcf cfMPI(argc, argv);

    if (argc > 1) {
      pressureLambda = std::atof(argv[1]);
    }

    int nx = 9;
    int ny = 9;
    int nz = 9;

    std::vector<double> ul2, curlul2, pl2, gradpl2, divl2, divmax;
    std::vector<double> h, convu, convcurlu, convp, convgradp;

    const int iters = 3;
    for (int i = 0; i < iters; ++i) {
      std::cout << "\n ------------------------------------- " << std::endl;
      std::cout << " --- 3D Kikuchi cut filled-sphere pressure-robustness test --- " << std::endl;
      std::cout << " lambda = " << pressureLambda << std::endl;

      Mesh Kh(nx, ny, nz, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0);
      const R hi = 2.0/(nx - 1);

      Space Uh_(Kh, DataFE<Mesh>::Ned0); // H(curl) velocity, lowest-order Nedelec
      Space Ph_(Kh, DataFE<Mesh>::P1);   // H1 pressure
      Space Lh_(Kh, DataFE<Mesh>::P1);   // level-set interpolation space

      Fun_h levelSet(Lh_, fun_levelSet);
      InterfaceLevelSet<Mesh> interface(Kh, levelSet);

      ActiveMesh<Mesh> Khi(Kh);
      // fun_levelSet = r - R is positive outside; truncate sign +1 removes the exterior.
      Khi.truncate(interface, 1);
      Khi.info();

      MacroElement<Mesh> macro(Khi, 0.25);

      CutSpace Uh(Khi, Uh_);
      CutSpace Ph(Khi, Ph_);

      // Pressure-robust load handling: H(curl) interpolation before addLinear.
      Fun_h fh(Uh, fun_rhs);

      CutFEM<Mesh> stokes(Uh);
      stokes.add(Ph);

      FunTest u(Uh, 3, 0), v(Uh, 3, 0);
      FunTest p(Ph, 1, 0), q(Ph, 1, 0);

      const R etaGhost = 1.0;

      std::cout << "Assembling..." << std::endl;
      stokes.addBilinear(
        + innerProduct(curl(u), curl(v))
        + innerProduct(grad(p), v)
        , Khi
      );
      stokes.addLinear(
        + innerProduct(fh.exprList(), v)
        , Khi
      );
      stokes.addBilinear(
        + innerProduct(u, grad(q))
        , Khi
      );

      // Ghost-product stabilisation 
      stokes.addFaceStabilization(
        + innerProduct(etaGhost * pow(hi,1) * jump(curl(u)), jump(curl(v)))
        + innerProduct(etaGhost * pow(hi,1) * jump(grad(p)), jump(v))
        + innerProduct(etaGhost * pow(hi,1) * jump(u), jump(grad(q)))
        // + innerProduct(etaGhost * pow(hi,1) * jump(p), jump(q))
        // + innerProduct(etaGhost * pow(hi,3) * jump(grad(p)), jump(grad(q)))
        , Khi
        , macro
      );

      // DEBUG block 1
      // {
      // auto print_vec_stats = [](const Rn &x, const std::string &name) {
      //     R linf = 0.;
      //     R l2   = 0.;
      //     R l1   = 0.;
      //     int imax = -1;

      //     for (int i = 0; i < x.size(); ++i) {
      //         R a = std::abs(x(i));
      //         l1 += a;
      //         l2 += x(i) * x(i);
      //         if (a > linf) {
      //             linf = a;
      //             imax = i;
      //         }
      //     }

      //     std::cout << name
      //               << " size=" << x.size()
      //               << " l1=" << l1
      //               << " l2=" << std::sqrt(l2)
      //               << " linf=" << linf
      //               << " imax=" << imax
      //               << std::endl;
      // };
      // Rn rhs_before_ghost(stokes.rhs_);
      // }

      // RHS ghost product contribution: s_h(f_h, v_h)
      stokes.addFaceStabilizationRHS(
        // + innerProduct(jump(fh, 1., -1.), etaGhost * pow(hi,1) *jump(v))
        + innerProduct(jump(fh.exprList()), etaGhost * pow(hi,1) *jump(v))
        , Khi
        , macro
      );

      // DEBUG block 2
      // {
      // Rn rhs_after_ghost(stokes.rhs_);
      // Rn rhs_ghost(rhs_after_ghost);
      // rhs_ghost -= rhs_before_ghost;

      // print_vec_stats(rhs_before_ghost, "rhs before ghost");
      // print_vec_stats(rhs_after_ghost,  "rhs after ghost ");
      // print_vec_stats(rhs_ghost,        "rhs ghost delta ");
      // int nb_u = Uh.get_nb_dof();
      // int nb_p = Ph.get_nb_dof();

      // Rn rhs_ghost_u(nb_u);
      // Rn rhs_ghost_p(nb_p);

      // for (int j = 0; j < nb_u; ++j)
      //     rhs_ghost_u(j) = rhs_ghost(j);

      // for (int j = 0; j < nb_p; ++j)
      //     rhs_ghost_p(j) = rhs_ghost(nb_u + j);

      // print_vec_stats(rhs_ghost_u, "rhs ghost delta, velocity block");
      // print_vec_stats(rhs_ghost_p, "rhs ghost delta, pressure block");
      // }
      
      // p_exact has zero mean over the exact ball
      stokes.addLagrangeMultiplier(
        innerProduct(1, p), 0.0
        , Khi
      );

      std::cout << "Solving..." << std::endl;
      stokes.solve("mumps");

      const int nb_vel_dof  = Uh.get_nb_dof();
      const int nb_pres_dof = Ph.get_nb_dof();

      std::cout << "Lagrange multiplier value: " << std::endl;
      std::cout << stokes.rhs_(nb_vel_dof + nb_pres_dof) << std::endl;

      Rn_ data_uh = stokes.rhs_(SubArray(nb_vel_dof, 0));
      Rn_ data_ph = stokes.rhs_(SubArray(nb_pres_dof, nb_vel_dof));

      Fun_h uh(Uh, data_uh);
      Fun_h ph(Ph, data_ph);

      auto uh_0dx = dx(uh.expr(0));
      auto uh_0dy = dy(uh.expr(0));
      auto uh_0dz = dz(uh.expr(0));
      auto uh_1dx = dx(uh.expr(1));
      auto uh_1dy = dy(uh.expr(1));
      auto uh_1dz = dz(uh.expr(1));
      auto uh_2dx = dx(uh.expr(2));
      auto uh_2dy = dy(uh.expr(2));
      auto uh_2dz = dz(uh.expr(2));

      auto curl_uh_0 = uh_2dy - uh_1dz;
      auto curl_uh_1 = uh_0dz - uh_2dx;
      auto curl_uh_2 = uh_1dx - uh_0dy;

      auto ph_dx = dx(ph.expr(0));
      auto ph_dy = dy(ph.expr(0));
      auto ph_dz = dz(ph.expr(0));

      {
        Fun_h solp(Ph, fun_exact_p);
        Fun_h solpErr(Ph, fun_exact_p);
        solpErr.v -= ph.v;
        solpErr.v.map(fabs);

        Paraview<Mesh> writer(Khi, "stokes3D_kikuchi_sphere_presrob_" + std::to_string(i) + ".vtk");
        writer.add(uh, "velocity", 0, 3);
        writer.add(ph, "pressure", 0, 1);
        writer.add(fh, "rhs_Hcurl", 0, 3);
        writer.add(curl_uh_0, "curl_u_0");
        writer.add(curl_uh_1, "curl_u_1");
        writer.add(curl_uh_2, "curl_u_2");
        writer.add(uh_0dx + uh_1dy + uh_2dz, "divergence");
        writer.add(solp, "pressureExact", 0, 1);
        writer.add(solpErr, "pressureError", 0, 1);
      }

      const R errU = L2normCut(uh, fun_exact_u, 0, 3);
      const R errCurlU = std::sqrt(
        integral(Khi, curl_uh_0*curl_uh_0 + curl_uh_1*curl_uh_1 + curl_uh_2*curl_uh_2, 0)
      );
      const R errP = L2normCut(ph, fun_exact_p, 0, 1);
      const R errGradP = std::sqrt(
        std::pow(L2normCut(ph_dx, fun_grad_p_0, Khi), 2)
        + std::pow(L2normCut(ph_dy, fun_grad_p_1, Khi), 2)
        + std::pow(L2normCut(ph_dz, fun_grad_p_2, Khi), 2)
      );
      const R errDiv = L2normCut(uh_0dx + uh_1dy + uh_2dz, Khi);
      const R maxErrDiv = maxNormCut(uh_0dx + uh_1dy + uh_2dz, Khi);

      h.push_back(hi);
      ul2.push_back(errU);
      curlul2.push_back(errCurlU);
      pl2.push_back(errP);
      gradpl2.push_back(errGradP);
      divl2.push_back(errDiv);
      divmax.push_back(maxErrDiv);

      if (i == 0) {
        convu.push_back(0.0);
        convcurlu.push_back(0.0);
        convp.push_back(0.0);
        convgradp.push_back(0.0);
      } else {
        convu.push_back(std::log(ul2[i]/ul2[i-1])/std::log(h[i]/h[i-1]));
        convcurlu.push_back(std::log(curlul2[i]/curlul2[i-1])/std::log(h[i]/h[i-1]));
        convp.push_back(std::log(pl2[i]/pl2[i-1])/std::log(h[i]/h[i-1]));
        convgradp.push_back(std::log(gradpl2[i]/gradpl2[i-1])/std::log(h[i]/h[i-1]));
      }

      nx = 2*nx - 1;
      ny = 2*ny - 1;
      nz = 2*nz - 1;
    }

    std::cout << "\n" << std::left
      << std::setw(10) << std::setfill(' ') << "h"
      << std::setw(15) << std::setfill(' ') << "err u"
      << std::setw(15) << std::setfill(' ') << "conv u"
      << std::setw(15) << std::setfill(' ') << "err curlu"
      << std::setw(15) << std::setfill(' ') << "conv curlu"
      << std::setw(15) << std::setfill(' ') << "err p"
      << std::setw(15) << std::setfill(' ') << "conv p"
      << std::setw(15) << std::setfill(' ') << "err gradp"
      << std::setw(15) << std::setfill(' ') << "conv gradp"
      << std::setw(15) << std::setfill(' ') << "err divu"
      << std::setw(15) << std::setfill(' ') << "err maxdivu"
      << "\n" << std::endl;

    for (int i = 0; i < h.size(); ++i) {
      std::cout << std::left
        << std::setw(10) << std::setfill(' ') << h[i]
        << std::setw(15) << std::setfill(' ') << ul2[i]
        << std::setw(15) << std::setfill(' ') << convu[i]
        << std::setw(15) << std::setfill(' ') << curlul2[i]
        << std::setw(15) << std::setfill(' ') << convcurlu[i]
        << std::setw(15) << std::setfill(' ') << pl2[i]
        << std::setw(15) << std::setfill(' ') << convp[i]
        << std::setw(15) << std::setfill(' ') << gradpl2[i]
        << std::setw(15) << std::setfill(' ') << convgradp[i]
        << std::setw(15) << std::setfill(' ') << divl2[i]
        << std::setw(15) << std::setfill(' ') << divmax[i]
        << std::endl;
    }

    std::cout << "CPU time = " << CPUtime() - cpubegin << std::endl;
  }
#endif

