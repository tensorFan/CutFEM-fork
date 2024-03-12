#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
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


#define PROBLEM_UNFITTED_PRESROB2_STOKES_VORTICITY_4FIELD // (2023 autumn)
// #define PROBLEM_UNFITTED_HANSBO_STOKES_VORTICITY_4FIELD // (2023 autumn)




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
      // stokes.addFaceStabilization( // [previously h^(2k+1) + macro]
      //   -innerProduct(pPenParam*pow(hi,0)*jump(p_itf), jump(q_itf))
      //   -innerProduct(pPenParam*pow(hi,2)*jump(grad(p_itf)*n), jump(grad(q_itf)*n)) 
      // , Kh_itf
      // // , macro_itf // somehow fails at last iteration when not using macro (due to umfpack maybe?)
      // );
      // stokes.addBilinear( 
      //   -innerProduct(pPenParam*pow(hi,1)*grad(p_itf)*n, grad(q_itf)*n) 
      // , Kh_itf
      // );
      // [Saras test (12/03/24):]
      stokes.addFaceStabilization( // [previously h^(2k+1) + macro]
        -innerProduct(pPenParam*pow(hi,1)*jump(p_itf), jump(q_itf))
        -innerProduct(pPenParam*pow(hi,3)*jump(grad(p_itf)*n), jump(grad(q_itf)*n)) 
      , Kh_itf
      // , macro_itf // somehow fails at last iteration when not using macro (due to umfpack maybe?)
      );
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
      stokes.setDirichlet(u00, Khi);
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