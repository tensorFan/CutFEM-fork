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
#include "../num/DA.hpp"


// #include "../num/gnuplot.hpp"

// #define FITTED_DARCY_EXAMPLE_NOTCIRCLE

// #define DARCY_EXAMPLE_NOTCIRCLE // Ex 2
#define DARCY_EXAMPLE_CIRCLE // Ex 1
// #define DARCY_EXAMPLE_CIRCLE_SARA_TESTS
// #define DARCY_EXAMPLE_TWO_BC_ANNULUS_PLUS_INTERFACE // Ex 3
// #define DARCY_2FIELD_EXAMPLE_TWO_BC_ANNULUS_PLUS_INTERFACE

#ifdef FITTED_DARCY_EXAMPLE_NOTCIRCLE // ex 1
  namespace Data_Darcy_Square { // Example 1 of paper
    R d_x = 1.;
    R d_y = 0.5;
    // [divu = linear, DOMAIN = [0,1]x[0,0.5]]
    R fun_force(double *P, int compInd) {
      return 0;
    }
    R fun_div(double *P, int compInd) {// is also exact divergence
      R x = P[0]; R y = P[1];
      return 2*x+2*y-(1+0.5);
    }
    R fun_exact_u(double *P, int compInd) {
      R x = P[0]; R y =P[1];
      if (compInd == 0) return x*(x-1);
      else return y*(y-0.5);
    }
    R fun_exact_p(double *P, int compInd) {
      R x = P[0]; R y = P[1];
      return -(x*x*x/3-x*x/2 + y*y*y/3-0.5*y*y/2);
    }

    // [divu = const, DOMAIN = [0,1]^2]
    // R fun_force(const R2 P, int compInd, int dom) {
    //   return 0;
    // }
    // R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
    //   R x = P.x; R y = P.y;
    //   return 0;
    // }
    // R fun_exact_u(const R2 P, int compInd, int dom) {
    //   R x = P.x; R y = P.y;
    //   if (compInd == 0) return -x;
    //   if (compInd == 1) return y-1;
    // }
    // R fun_exact_p(const R2 P, int compInd, int dom) {
    //   R x = P.x; R y = P.y;
    //   return -(-x*x/2+(y-1)*(y-1)/2);
    // }
  }
  using namespace Data_Darcy_Square;

  int main(int argc, char** argv ) {
    typedef TestFunction<Mesh2> FunTest;
    typedef FunFEM<Mesh2> Fun_h;
    typedef Mesh2 Mesh;
    typedef ActiveMeshT2 CutMesh;
    typedef FESpace2   Space;
    typedef CutFESpaceT2 CutSpace;

    // MPIcf cfMPI(argc,argv);
    const double cpubegin = CPUtime();

    int nx =11; // 6
    int ny =11; // 6

    std::vector<double> hPr,uPrint,pPrint,divPrint,divPrintLoc,maxDivPrint,lagrPrint,convuPr,convpPr,convdivPr,convdivPrLoc,convmaxdivPr,convLagrPr;

    int iters = 5;
    for(int i=0; i<iters; ++i) {
        Mesh Kh(nx, ny, 0., 0., d_x, d_y);

        const R h = 1./(nx-1);
        const R invh = 1./h;

        Space W0h(Kh, DataFE<Mesh>::RT0); Space P0h(Kh, DataFE<Mesh>::P0); // for area computation (always lowest order)
        
        // Space V2h(Kh, DataFE<Mesh>::RT2); Space Q2h(Kh, DataFE<Mesh>::P2); // for the RHS
        Lagrange2 FEvelocity2(4); Space W2h(Kh, FEvelocity2); Space P2h(Kh, DataFE<Mesh>::P2); // for the RHS

        Space Wh(Kh, DataFE<Mesh>::RT0);
        Space Ph(Kh, DataFE<Mesh>::P0);
        // Space Qh_itf(Kh, DataFE<Mesh>::P2dc);

        Normal n;
        Tangent t;

        // Create FEM object
        FEM<Mesh2> darcy(Wh); darcy.add(Ph); //darcy.add(Ph_itf);

        // Fun_h fv(Wh, fun_force);
        Fun_h fq(P2h, fun_div);
        Fun_h p0(P2h, fun_exact_p);
        Fun_h u0(W2h, fun_exact_u);

        // [Params]
        FunTest p(Ph,1), q(Ph,1), u(Wh,2), v(Wh,2);
        // FunTest p_itf(Ph_itf,1), q_itf(Ph_itf,1);

        // [Area]
        FEM<Mesh2> area_comp(P0h); FunTest q0(P0h,1);
        area_comp.addLinear(
        +innerProduct(1, q0)
        , Kh
        ); 
        R area = area_comp.rhs_.sum();

        // [ASSEMBLY]
        darcy.addBilinear(
        +innerProduct(u, v)
        -innerProduct(p, div(v))
        +innerProduct(div(u), q)
        , Kh
        );
        darcy.addLinear(
        // innerProduct(fv.expression(2), v)
        +innerProduct(fq.exprList(), q)
        , Kh
        );

        // [Essential boundary conditions]
        Fun_h u00(Wh, fun_exact_u);
        darcy.setDirichlet(u00, Kh); 

        darcy.addLagrangeMultiplier(
        +innerProduct(1, p), 0
        , Kh
        );

        // [Natural BC]
        // darcy.addLinear(
        // -innerProduct(p0.expr(), v*n)
        // , Kh, INTEGRAL_BOUNDARY
        // );

        // [Add to last diagonal entry of LHS]
        // int nb_dof = Wh.get_nb_dof()+Ph.get_nb_dof()+Ph_itf.get_nb_dof();
        // darcy.mat_[make_pair(nb_dof,nb_dof)] = area;

        matlab::Export(darcy.mat_[0], "mat"+std::to_string(i)+".dat"); 
        darcy.solve("umfpack");

        std::cout << "Mesh " << i << " done" << std::endl;

        // EXTRACT SOLUTION
        int nbdof_vel = Wh.get_nb_dof();
        int nbdof_pres = Ph.get_nb_dof();
        Rn_ data_uh = darcy.rhs_(SubArray(nbdof_vel,0));
        Rn_ data_ph = darcy.rhs_(SubArray(nbdof_pres,nbdof_vel));

        Fun_h uh(Wh, data_uh);
        Fun_h ph(Ph, data_ph);
        // std::cout << data_ph_itf << std::endl;

        auto uh_0dx = dx(uh.expr(0));
        auto uh_1dy = dy(uh.expr(1));        

        // [Post process pressure]
        R meanP = integral(Kh,p0.expr(),0);
        R meanPfem = integral(Kh,ph.expr(),0);
        // std::cout << meanP << std::endl;
        ph.v -= meanPfem/area;
        ph.v += meanP/area;

        // L2 norm vel
        R errU      = L2norm(uh,fun_exact_u,0,2);
        R errP      = L2norm(ph,fun_exact_p,0,1);
        R errDiv    = L2norm(uh_0dx+uh_1dy,fun_div,Kh);
        R maxErrDiv = maxNorm(uh_0dx+uh_1dy-fq.expr(),Kh);
        
        // R errLagr = L2normCut(ph_itf,fun_exact_p,0,1);
        // R errLagr = L2normSurf(ph_itf,fun_exact_p_surf,interface,0,1);

        // [PLOTTING]
        {
        // Fun_h soluh(Wh, fun_exact_u);
        // Fun_h solph(Wh, fun_exact_p);

        // Fun_h divSolh(P2h, fun_div);
        // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

        Paraview<Mesh> writer(Kh, "darcyFict_"+std::to_string(i)+".vtk");
        writer.add(uh, "velocity" , 0, 2);
        writer.add(u0, "velocityExact" , 0, 2);
        writer.add(ph, "pressure" , 0, 1);
        writer.add(p0, "pressureExact" , 0, 1);
        writer.add(uh_0dx+uh_1dy, "divergence");
        // soluh.v -= uh.v; soluh.v.map(fabs);
        // writer.add(soluh, "velocityError" , 0, 2);
        writer.add(fabs((uh_0dx+uh_1dy)-fq.expr()), "divergenceError");
        // writer.add(ph_itf, "bdry pressure" , 0, 1);

        // Paraview<Mesh> writerMacro(Kh, "Kh" + to_string(i) + ".vtk");
        // writerMacro.add(levelSet, "levelSet.vtk", 0, 1);
        // writerMacro.writeActiveMesh(Kh_itf, "Kh_itf"+to_string(i) + ".vtk");
        // writerMacro.writeMacroInnerEdge(macro_itf, "macro_itf_internal_edge"+to_string(i) + ".vtk");
        // writerMacro.writeMacroOutterEdge(macro_itf, "macro_itf_outer_edge"+to_string(i) + ".vtk");
        // writerMacro.writeMacroElement(macro_itf, "macro_itf"+to_string(i) + ".vtk");

        }

        hPr.push_back(h);

        pPrint.push_back(errP);
        uPrint.push_back(errU);
        divPrint.push_back(errDiv);
        maxDivPrint.push_back(maxErrDiv);
        // lagrPrint.push_back(errLagr);

        if(i==0) {convpPr.push_back(0);convuPr.push_back(0);convdivPr.push_back(0);convdivPrLoc.push_back(0);convmaxdivPr.push_back(0);}//convLagrPr.push_back(0);}
        else {
        convpPr.push_back( log(pPrint[i]/pPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convuPr.push_back( log(uPrint[i]/uPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convdivPr.push_back( log(divPrint[i]/divPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convmaxdivPr.push_back( log(maxDivPrint[i]/maxDivPrint[i-1])/log(hPr[i]/hPr[i-1]));
        // convLagrPr.push_back( log(lagrPrint[i]/lagrPrint[i-1])/log(hPr[i]/hPr[i-1]));
        }

        nx = 2*nx-1;
        ny = 2*ny-1;
        // nx += 1;
        // ny += 1;
        // nx = (int)round( (1+0.2*i)*nx/2 )*2; // Makes a nonuniform refinement to an EVEN integer
        // ny = (int)round( (1+0.2*i)*ny/2 )*2;
        // std::cout << nx << std::endl;
        // shift = 0.5+(i+1)*h_i/iters; // moves one grid cell over entire span
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
    // << std::setw(15) << std::setfill(' ') << "err p_itf"
    // << std::setw(15) << std::setfill(' ') << "conv p_itf"    
    << "\n" << std::endl;
    for(int i=0;i<uPrint.size();++i) {
        std::cout << std::left
        << std::setw(10) << std::setfill(' ') << hPr[i]
        << std::setw(15) << std::setfill(' ') << pPrint[i]
        << std::setw(15) << std::setfill(' ') << convpPr[i]
        << std::setw(15) << std::setfill(' ') << uPrint[i]
        << std::setw(15) << std::setfill(' ') << convuPr[i]
        << std::setw(15) << std::setfill(' ') << divPrint[i]
        // << std::setw(15) << std::setfill(' ') << convdivPr[i]
        << std::setw(15) << std::setfill(' ') << maxDivPrint[i]
        // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
        // << std::setw(15) << std::setfill(' ') << lagrPrint[i]
        // << std::setw(15) << std::setfill(' ') << convLagrPr[i]
        << std::endl;
    }
  }

#endif

#ifdef DARCY_EXAMPLE_NOTCIRCLE // ex 2
  namespace Data_Darcy_Square { // Example 1 of paper
    R d_x = 1.;
    R d_y = 0.5;
    R fun_levelSet(const R2 P, const int i) {
      return 0.5 - P.y; //0.5+1e-14 - P.y;
    }
    // R fun_levelSet(const R2 P, const int i) {
    //   return 0.5-pow(P.x-0.5,4)-P.y;
    // }
    // [divu = linear, DOMAIN = [0,1]x[0,0.5]]
    R fun_force(const R2 P, int compInd, int dom) {
      return 0;
    }
    R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
      R x = P.x; R y = P.y;
      return 2*x+2*y-(1+0.5);
    }
    R fun_exact_u(const R2 P, int compInd, int dom) {
      R x = P.x; R y =P.y;
      if (compInd == 0) return x*(x-1);
      else return y*(y-0.5);
    }
    R fun_exact_p(const R2 P, int compInd, int dom) {
      R x = P.x; R y = P.y;
      return -(x*x*x/3-x*x/2 + y*y*y/3-0.5*y*y/2);
    }
    R fun_exact_p_surf(const R2 P, int compInd) {
      R x = P.x; R y = P.y;
      return -(x*x*x/3-x*x/2 + y*y*y/3-0.5*y*y/2);
    }

    // [divu = const, DOMAIN = [0,1]^2]
    // R fun_force(const R2 P, int compInd, int dom) {
    //   return 0;
    // }
    // R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
    //   R x = P.x; R y = P.y;
    //   return 0;
    // }
    // R fun_exact_u(const R2 P, int compInd, int dom) {
    //   R x = P.x; R y = P.y;
    //   if (compInd == 0) return -x;
    //   if (compInd == 1) return y-1;
    // }
    // R fun_exact_p(const R2 P, int compInd, int dom) {
    //   R x = P.x; R y = P.y;
    //   return -(-x*x/2+(y-1)*(y-1)/2);
    // }
  }
  using namespace Data_Darcy_Square;

  int main(int argc, char** argv ) {
    typedef TestFunction<Mesh2> FunTest;
    typedef FunFEM<Mesh2> Fun_h;
    typedef Mesh2 Mesh;
    typedef ActiveMeshT2 CutMesh;
    typedef FESpace2   Space;
    typedef CutFESpaceT2 CutSpace;

    // MPIcf cfMPI(argc,argv);
    const double cpubegin = CPUtime();

    int nx =11; // 6
    int ny =11; // 6
    // Kh_init0.truncate(interface_init, -1);

    std::vector<double> hPr,uPrint,pPrint,divPrint,divPrintLoc,maxDivPrint,lagrPrint,convuPr,convpPr,convdivPr,convdivPrLoc,convmaxdivPr,convLagrPr;

    int iters = 4;
    for(int i=0; i<iters; ++i) {
        Mesh Kh(nx, ny, 0., 0., d_x, d_y+globalVariable::Epsilon);

        const R h = 1./(nx-1);
        const R invh = 1./h;

        Space V0h(Kh, DataFE<Mesh>::RT0); Space Q0h(Kh, DataFE<Mesh>::P0); // for area computation (always lowest order)
        
        // Space V2h(Kh, DataFE<Mesh>::RT2); Space Q2h(Kh, DataFE<Mesh>::P2); // for the RHS
        Lagrange2 FEvelocity2(4); Space V2h(Kh, FEvelocity2); Space Q2h(Kh, DataFE<Mesh>::P2); // for the RHS

        Space Vh(Kh, DataFE<Mesh>::RT1);
        Space Qh(Kh, DataFE<Mesh>::P1dc);
        Space Qh_itf(Kh, DataFE<Mesh>::P2dc);

        Space Lh(Kh, DataFE<Mesh2>::P1);
        Fun_h levelSet(Lh, fun_levelSet);
        InterfaceLevelSet<Mesh> interface(Kh, levelSet);
        Normal n;
        Tangent t;

        // MAIN MESH
        ActiveMesh<Mesh> Kh_i(Kh);       // Kh_i.info();
        Kh_i.truncate(interface, -1); // [-1 removes the negative part]

        CutSpace W0h(Kh_i, V0h); CutSpace P0h(Kh_i, Q0h);
        CutSpace W2h(Kh_i, V2h); CutSpace P2h(Kh_i, Q2h);

        CutSpace Wh(Kh_i, Vh);// Wh.info();
        CutSpace Ph(Kh_i, Qh);

        // MacroElement<Mesh> macro(Kh_i, 1);
        
        // SURFACE MESH
        ActiveMesh<Mesh> Kh_itf(Kh);
        Kh_itf.createSurfaceMesh(interface);
        CutSpace Ph_itf(Kh_itf, Qh_itf);
        // MacroElementSurface<Mesh> macro_itf(Kh_itf, interface, 0.3); // 1

        // Create CutFEM object
        CutFEM<Mesh2> darcy(Wh); darcy.add(Ph); darcy.add(Ph_itf);

        // Fun_h fv(Wh, fun_force);
        Fun_h fq(P2h, fun_div);
        Fun_h p0(P2h, fun_exact_p);
        Fun_h u0(W2h, fun_exact_u);

        // [Params]
        FunTest p(Ph,1), q(Ph,1), u(Wh,2), v(Wh,2);
        FunTest p_itf(Ph_itf,1), q_itf(Ph_itf,1);

        // [Area]
        CutFEM<Mesh2> area_comp(P0h); FunTest q0(P0h,1);
        area_comp.addLinear(
        +innerProduct(1, q0)
        , Kh_i
        ); 
        R area = area_comp.rhs_.sum();

        // [ASSEMBLY]
        darcy.addBilinear(
        +innerProduct(u, v)
        -innerProduct(p, div(v))
        +innerProduct(div(u), q)
        , Kh_i
        );
        darcy.addLinear(
        // innerProduct(fv.expression(2), v)
        +innerProduct(fq.exprList(), q)
        , Kh_i
        );


        // [GHOST PENALTY]
        double uPenParam = 1e0; // 1e0 
        double pPenParam = 1e0; // 1e0
        double itfPenParam = 1e0; // 1e-2
        darcy.addPatchStabilization( // [h^(2k) h^(2k)]
        +innerProduct(uPenParam*jump(u), jump(v)) 
        -innerProduct(pPenParam*jump(p), jump(div(v)))
        +innerProduct(pPenParam*jump(div(u)), jump(q))
        , Kh_i
        // , macro
        );
        FunTest grad2pitfn = grad(grad(p_itf)*n)*n;
        FunTest grad2pitf = grad(grad(p_itf));
        darcy.addFaceStabilization(
        -innerProduct(itfPenParam*pow(h,-1)*jump(p_itf), jump(q_itf))
        -innerProduct(itfPenParam*pow(h,1)*jump(grad(p_itf)), jump(grad(q_itf))) 
        -innerProduct(itfPenParam*pow(h,3)*jump(grad2pitf), jump(grad2pitf))
        , Kh_itf
        // , macro_itf
        );
        darcy.addBilinear(
        // -innerProduct(itfPenParam*pow(h,-1)*(p_itf), (q_itf))
        -innerProduct(itfPenParam*pow(h,1)*(grad(p_itf)*n), (grad(q_itf)*n)) 
        -innerProduct(itfPenParam*pow(h,3)*(grad2pitfn), (grad2pitfn))
        , interface
        );
        darcy.addBilinearIntersection(
        // -innerProduct(itfPenParam*pow(h,-1)*(p_itf), (q_itf))
        -innerProduct(itfPenParam*pow(h,1)*(grad(p_itf)*n), (grad(q_itf)*n)) 
        -innerProduct(itfPenParam*pow(h,3)*(grad2pitfn), (grad2pitfn))
        , Kh_itf, Kh_i, INTEGRAL_BOUNDARY
        );
        // darcy.BaseFEM::addBilinear( 
        //   -innerProduct(itfPenParam*pow(h,0)*grad(p_itf)*n, grad(q_itf)*n) 
        // , Kh_itf//, INTEGRAL_EXTENSION, 1
        // );

        // [Boundary conditions]
        darcy.addBilinear(
        +innerProduct(p_itf, v*n) 
        +innerProduct(u*n, q_itf)
        , interface
        );
        darcy.addLinear(fun_exact_u,
        +innerProduct(1, q_itf*n)
        , interface
        ); 
        darcy.addBilinearIntersection(
        +innerProduct(p_itf, v*n) 
        +innerProduct(u*n, q_itf)
        , Kh_itf, Kh_i, INTEGRAL_BOUNDARY
        );
        darcy.addLinearIntersection(
        +innerProduct(u0*n, q_itf)
        , Kh_itf, Kh_i, INTEGRAL_BOUNDARY
        );

        Fun_h u00(Wh, fun_exact_u);
        darcy.setDirichletHdiv(u00, Kh_i); 

        // [LAGRANGE MULT]
        darcy.addLagrangeMultiplier(
        +innerProduct(1, p), 0
        , Kh_i
        );

        // [Add to last diagonal entry of LHS]
        // int nb_dof = Wh.get_nb_dof()+Ph.get_nb_dof()+Ph_itf.get_nb_dof();
        // darcy.mat_[make_pair(nb_dof,nb_dof)] = area;

        matlab::Export(darcy.mat_[0], "mat"+std::to_string(i)+"Cut.dat"); 
        // nx = 2*nx-1;
        // ny = 2*ny-1;
        // continue;
        darcy.solve("umfpack");

        std::cout << "Lagrange multiplier value: " << std::endl;
        std::cout << darcy.rhs_(Wh.get_nb_dof()+Ph.get_nb_dof()+Ph_itf.get_nb_dof())<< std::endl;

        // EXTRACT SOLUTION
        int nbdof_vel = Wh.get_nb_dof();
        int nbdof_pres = Ph.get_nb_dof();
        int nbdof_pres_itf = Ph_itf.get_nb_dof();
        Rn_ data_uh = darcy.rhs_(SubArray(nbdof_vel,0));
        Rn_ data_ph = darcy.rhs_(SubArray(nbdof_pres,nbdof_vel));
        Rn_ data_ph_itf = darcy.rhs_(SubArray(nbdof_pres_itf,nbdof_pres+nbdof_vel));

        Fun_h uh(Wh, data_uh);
        Fun_h ph(Ph, data_ph);
        Fun_h ph_itf(Ph_itf, data_ph_itf);
        // std::cout << data_ph_itf << std::endl;

        auto uh_0dx = dx(uh.expr(0));
        auto uh_1dy = dy(uh.expr(1));        

        // [Post process pressure]
        R meanP = integral(Kh_i,p0.expr(),0);
        R meanPfem = integral(Kh_i,ph.expr(),0);
        // std::cout << meanP << std::endl;
        ph.v -= meanPfem/area;
        ph.v += meanP/area;

        // L2 norm vel
        R errU      = L2normCut(uh,fun_exact_u,0,2);
        R errP      = L2normCut(ph,fun_exact_p,0,1);
        R errDiv    = L2normCut (uh_0dx+uh_1dy,fun_div,Kh_i);
        R maxErrDiv = maxNormCut(uh_0dx+uh_1dy,fun_div,Kh_i);
        // R errLagr = L2normCut(ph_itf,fun_exact_p,0,1);
        R errLagr = L2normSurf(ph_itf,fun_exact_p_surf,interface,0,1);

        // [PLOTTING]
        {
        // Fun_h soluh(Wh, fun_exact_u);
        // Fun_h solph(Wh, fun_exact_p);

        // Fun_h divSolh(P2h, fun_div);
        // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

        Paraview<Mesh> writer(Kh_i, "darcyFict_"+std::to_string(i)+".vtk");
        writer.add(uh, "velocity" , 0, 2);
        writer.add(u0, "velocityExact" , 0, 2);
        writer.add(ph, "pressure" , 0, 1);
        writer.add(p0, "pressureExact" , 0, 1);
        writer.add(uh_0dx+uh_1dy, "divergence");
        // soluh.v -= uh.v; soluh.v.map(fabs);
        // writer.add(soluh, "velocityError" , 0, 2);
        writer.add(fabs((uh_0dx+uh_1dy)-fq.expr()), "divergenceError");
        // writer.add(ph_itf, "bdry pressure" , 0, 1);

        // Paraview<Mesh> writerMacro(Kh, "Kh" + to_string(i) + ".vtk");
        // writerMacro.add(levelSet, "levelSet.vtk", 0, 1);
        // writerMacro.writeActiveMesh(Kh_itf, "Kh_itf"+to_string(i) + ".vtk");
        // writerMacro.writeMacroInnerEdge(macro_itf, "macro_itf_internal_edge"+to_string(i) + ".vtk");
        // writerMacro.writeMacroOutterEdge(macro_itf, "macro_itf_outer_edge"+to_string(i) + ".vtk");
        // writerMacro.writeMacroElement(macro_itf, "macro_itf"+to_string(i) + ".vtk");

        }

        hPr.push_back(h);

        pPrint.push_back(errP);
        uPrint.push_back(errU);
        divPrint.push_back(errDiv);
        maxDivPrint.push_back(maxErrDiv);
        lagrPrint.push_back(errLagr);

        if(i==0) {convpPr.push_back(0);convuPr.push_back(0);convdivPr.push_back(0);convdivPrLoc.push_back(0);convmaxdivPr.push_back(0);convLagrPr.push_back(0);}
        else {
        convpPr.push_back( log(pPrint[i]/pPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convuPr.push_back( log(uPrint[i]/uPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convdivPr.push_back( log(divPrint[i]/divPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convmaxdivPr.push_back( log(maxDivPrint[i]/maxDivPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convLagrPr.push_back( log(lagrPrint[i]/lagrPrint[i-1])/log(hPr[i]/hPr[i-1]));
        }

        nx = 2*nx-1;
        ny = 2*ny-1;
        // nx += 1;
        // ny += 1;
        // nx = (int)round( (1+0.2*i)*nx/2 )*2; // Makes a nonuniform refinement to an EVEN integer
        // ny = (int)round( (1+0.2*i)*ny/2 )*2;
        // std::cout << nx << std::endl;
        // shift = 0.5+(i+1)*h_i/iters; // moves one grid cell over entire span
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
    << std::setw(15) << std::setfill(' ') << "err p_itf"
    << std::setw(15) << std::setfill(' ') << "conv p_itf"    
    << "\n" << std::endl;
    for(int i=0;i<uPrint.size();++i) {
        std::cout << std::left
        << std::setw(10) << std::setfill(' ') << hPr[i]
        << std::setw(15) << std::setfill(' ') << pPrint[i]
        << std::setw(15) << std::setfill(' ') << convpPr[i]
        << std::setw(15) << std::setfill(' ') << uPrint[i]
        << std::setw(15) << std::setfill(' ') << convuPr[i]
        << std::setw(15) << std::setfill(' ') << divPrint[i]
        // << std::setw(15) << std::setfill(' ') << convdivPr[i]
        // << std::setw(15) << std::setfill(' ') << divPrintLoc[i]
        << std::setw(15) << std::setfill(' ') << maxDivPrint[i]
        // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
        << std::setw(15) << std::setfill(' ') << lagrPrint[i]
        << std::setw(15) << std::setfill(' ') << convLagrPr[i]
        << std::endl;
    }
  }

#endif

#ifdef DARCY_EXAMPLE_CIRCLE // ex 1
  namespace Data_Circle_ZeroVel {
    R d_x = 1.;
    R d_y = 1.;
    R pie = M_PI;
    R shift = 0.5;
    R RAD = 0.45;
    R lambda = 1e2;
    R fun_levelSet(const R2 P, const int i) {
      return RAD - sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift));
    } 
    R fun_normal(const R2 P, const int i) {
      if (i==0) return (P.x-shift)*(P.x-shift)/sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift));
      else return (P.y-shift)*(P.y-shift)/sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift));
    }

    R fun_force(const R2 P, int compInd, int dom) {
      R x = P.x; R y = P.y;
      if (compInd == 0) return 0;
      else return lambda*(1-y+3*y*y);
    }
    R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
      return 0;
    }
    R fun_exact_u(const R2 P, int compInd, int dom) { // = -grad p + f
      R x = P.x; R y = P.y;
      if (compInd == 0) return 0;
      else return 0; // = -1e12*(3y*y-y+1)+fy
    }
    R fun_exact_p(const R2 P, int compInd, int dom) {
      R x = P.x; R y = P.y;
      return lambda*(pow(y,3)-y*y/2+y-7/12);
    }
    R fun_exact_p_surf(const R2 P, int compInd) {
      R x = P.x; R y = P.y;
      return lambda*(pow(y,3)-y*y/2+y-7/12);
    }

    // wild boundary variation
    // R fun_force(const R2 P, int compInd, int dom) {
    //   R x = P.x; R y = P.y;
    //   R r = sqrt((x-shift)*(x-shift) + (y-shift)*(y-shift));
    //   if (compInd == 0) return sin(1.0/pow(RAD+0.1-r,1.6)*r);
    //   else return -sin(1.0/pow(RAD+0.1-r,1.6)*r);
    // }
    // R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
    //   double x = P.x;
    //   double y = P.y;
    //   double r = std::sqrt((x - shift) * (x - shift) + (y - shift) * (y - shift));
    //   double denominator = std::pow(0.1 + RAD - r, 1.6);
    //   double common_term = std::cos(r / denominator);

    //   // Compute partial derivatives
    //   double df_dx = (1.6 * (x - shift) / std::pow(0.1 + RAD - r, 2.6) + (x - shift) / (r * denominator)) * common_term;
    //   double df_dy = -(1.6 * (y - shift) / std::pow(0.1 + RAD - r, 2.6) + (y - shift) / (r * denominator)) * common_term;

    //   // Sum the components to get the divergence
    //   return df_dx + df_dy;
    // }
    // R fun_exact_u(const R2 P, int compInd, int dom) {
    //   return fun_force(P, compInd, dom);
    // }
    // R fun_exact_p(const R2 P, int compInd, int dom) {
    //   R x = P.x; R y = P.y;
    //   return 1;
    // }
    // R fun_exact_p_surf(const R2 P, int compInd) {
    //   return 1;
    // }

    // u=0
    // R fun_force(const R2 P, int compInd, int dom) {
    //   return 0;
    // }
    // R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
    //   R x = P.x; R y = P.y;
    //   return 0;
    // }
    // R fun_exact_u(const R2 P, int compInd, int dom) {
    //   R x = P.x; R y = P.y;
    //   if (compInd == 0) return 0;
    //   if (compInd == 1) return 0;
    //   else return 0;
    // }
    // R fun_exact_p(const R2 P, int compInd, int dom) {
    //   R x = P.x; R y = P.y;
    //   return 1e12;
    // }
    // R fun_exact_p_surf(const R2 P, int compInd) {
    //   return 1e12;
    // }
  }
  namespace Data_Darcy_Circle {
    R d_x = 1.;
    R d_y = 1.;
    R pie = M_PI;
    R shift = 0.5;

    R fun_levelSet(const R2 P, const int i) {
      return 0.45 - sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift));
    } 
    R fun_normal(const R2 P, const int i) {
      if (i==0) return (P.x-shift)*(P.x-shift)/sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift));
      else return (P.y-shift)*(P.y-shift)/sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift));
    }

    // // [u+grad p = 0]
    R fun_force(const R2 P, int compInd) {
      return 0;
    }
    R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
      return -8*pie*pie*sin(2*pie*P.x)*cos(2*pie*P.y);
    }
    R fun_exact_u(const R2 P, int compInd, int dom) {
      R x = P.x;
      R y = P.y;
      if (compInd==0) {
        return 2*pie*cos(2*pie*x)*cos(2*pie*y);
      } else {
        return -2*pie*sin(2*pie*x)*sin(2*pie*y);
      }
    }
    R fun_exact_p(const R2 P, int compInd, int dom) {
      return -sin(2*pie*P.x)*cos(2*pie*P.y);
    }
    R fun_exact_p_surf(const R2 P, int compInd) {
      return -sin(2*pie*P.x)*cos(2*pie*P.y);
    }

    // [divu = 0, DOMAIN = CIRCLE WITH RADIUS 0.25 AT (0.5,0.5) ]
    // R fun_levelSet(const R2 P, const int i) {
    //   return 0.25 - sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift));
    // } // normal = 4*(x-1/2,y-1/2)
    // R fun_force(const R2 P, int compInd) {
    //   return 0;
    // }
    // R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
    //   return 0;
    // }
    // R fun_exact_u(const R2 P, int compInd, int dom) {
    //   if (compInd==0) {
    //     return cos(P.x)*sinh(P.y);
    //   } else {
    //     return sin(P.x)*cosh(P.y);
    //   }
    // }
    // R fun_exact_p(const R2 P, int compInd, int dom) {
    //   return -sin(P.x)*sinh(P.y) - (cos(1) - 1)*(cosh(1) - 1);
    // }

    // [u is linear, int_bdry u*n = pi/8]
    // R fun_force(const R2 P, int compInd) {
    //   return 0;
    // }
    // R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
    //   return 2;
    // }
    // R fun_exact_u(const R2 P, int compInd, int dom) {
    //   if (compInd==0) {
    //     return P.x-0.5;
    //   } else {
    //     return P.y-0.5;
    //   }
    // }
    // R fun_exact_p(const R2 P, int compInd, int dom) {
    //   return -0.5*(P.x*P.x-P.x + P.y*P.y-P.y);
    // }

  }
  // using namespace Data_Circle_ZeroVel;
  using namespace Data_Darcy_Circle;

  int main(int argc, char** argv ) {
    typedef TestFunction<Mesh2> FunTest;
    typedef FunFEM<Mesh2> Fun_h;
    typedef Mesh2 Mesh;
    typedef ActiveMeshT2 CutMesh;
    typedef FESpace2   Space;
    typedef CutFESpaceT2 CutSpace;

    MPIcf cfMPI(argc,argv);
    const double cpubegin = CPUtime();

    int nx =11; // 6
    int ny =11; // 6
    // Kh_init0.truncate(interface_init, -1);

    std::vector<double> hPr,uPrint,pPrint,divPrint,divPrintLoc,maxDivPrint,lagrPrint,convuPr,convpPr,convdivPr,convdivPrLoc,convmaxdivPr,convLagrPr;

    int iters = 3;
    for(int i=0; i<iters; ++i) {
        Mesh Kh(nx, ny, 0., 0., d_x+globalVariable::Epsilon, d_y+globalVariable::Epsilon);

        const R h = 1./(nx-1);
        const R invh = 1./h;

        Space V0h(Kh, DataFE<Mesh>::RT0); Space Q0h(Kh, DataFE<Mesh>::P0); // for area computation (always lowest order)
        
        // Space V2h(Kh, DataFE<Mesh>::RT2); Space Q2h(Kh, DataFE<Mesh>::P2); // for the RHS
        Lagrange2 FEvelocity2(4); Space V2h(Kh, FEvelocity2); Space Q2h(Kh, DataFE<Mesh>::P2); // for the RHS

        Space Vh(Kh, DataFE<Mesh>::RT1);
        Space Qh(Kh, DataFE<Mesh>::P1dc);
        Space Qh_itf(Kh, DataFE<Mesh>::P1dc);

        Space Lh(Kh, DataFE<Mesh2>::P1);
        Fun_h levelSet(Lh, fun_levelSet);
        InterfaceLevelSet<Mesh> interface(Kh, levelSet);
        Normal n;
        Tangent t;

        // MAIN MESH
        ActiveMesh<Mesh> Kh_i(Kh);       // Kh_i.info();
        Kh_i.truncate(interface, -1); // [-1 removes the negative part]

        CutSpace W0h(Kh_i, V0h); CutSpace P0h(Kh_i, Q0h);
        CutSpace W2h(Kh_i, V2h); CutSpace P2h(Kh_i, Q2h);

        CutSpace Wh(Kh_i, Vh);// Wh.info();
        CutSpace Ph(Kh_i, Qh);

        // MacroElement<Mesh> macro(Kh_i, 1);
        
        // SURFACE MESH
        ActiveMesh<Mesh> Kh_itf(Kh);
        Kh_itf.createSurfaceMesh(interface);
        CutSpace Ph_itf(Kh_itf, Qh_itf);
        // MacroElementSurface<Mesh> macro_itf(Kh_itf, interface, 0.3); // 1

        // Create CutFEM object
        CutFEM<Mesh2> darcy(Wh); darcy.add(Ph); darcy.add(Ph_itf);

        Fun_h normal_phi(W2h, fun_normal);
        Fun_h fv(W2h, fun_force);
        Fun_h fq(P2h, fun_div);
        Fun_h p0(P2h, fun_exact_p);
        Fun_h u0(W2h, fun_exact_u);

        // [Params]
        FunTest p(Ph,1), q(Ph,1), u(Wh,2), v(Wh,2);
        FunTest p_itf(Ph_itf,1), q_itf(Ph_itf,1);

        // [Area]
        CutFEM<Mesh2> area_comp(P0h); FunTest q0(P0h,1);
        area_comp.addLinear(
        +innerProduct(1, q0)
        , Kh_i
        ); 
        R area = area_comp.rhs_.sum();

        // [ASSEMBLY]
        darcy.addBilinear(
        +innerProduct(u, v)
        -innerProduct(p, div(v))
        +innerProduct(div(u), q)
        , Kh_i
        );
        darcy.addLinear(
        +innerProduct(fv.exprList(), v)
        +innerProduct(fq.exprList(), q)
        , Kh_i
        );


        // [GHOST PENALTY]
        double uPenParam = 1e0; // 1e0 
        double pPenParam = 1e0; // 1e0
        double itfPenParam = 1e0; // 1e-2

        darcy.addPatchStabilization( // [h^(2k) h^(2k)]
        +innerProduct(uPenParam*jump(u), jump(v)) 
        -innerProduct(pPenParam*jump(p), jump(div(v)))
        +innerProduct(pPenParam*jump(div(u)), jump(q))
        , Kh_i
        // , macro
        );
        FunTest grad2pitfn = grad(grad(p_itf)*n)*n;
        FunTest grad2pitf = grad(grad(p_itf));
        darcy.addFaceStabilization(
        -innerProduct(itfPenParam*pow(h,-1)*jump(p_itf), jump(q_itf))
        -innerProduct(itfPenParam*pow(h,1)*jump(grad(p_itf)), jump(grad(q_itf))) 
        // -innerProduct(itfPenParam*pow(h,3)*jump(grad2pitf), jump(grad2pitf))
        , Kh_itf
        // , macro_itf
        );
        // darcy.addBilinear(
        //   // -innerProduct(itfPenParam*pow(h,-1)*p_itf, q_itf) 
        //   -innerProduct(itfPenParam*pow(h,1)*(grad(p_itf)*n), (grad(q_itf)*n)) 
        //   -innerProduct(itfPenParam*pow(h,3)*(grad2pitfn), (grad2pitfn))
        // , interface
        // );
        // darcy.BaseFEM::addBilinear(
        // -innerProduct(itfPenParam*pow(h,-2)*p_itf, q_itf) 
        // // -innerProduct(itfPenParam*pow(h,1)*grad(p_itf)*n, grad(q_itf)*n) 
        // , Kh_itf
        // );
        darcy.BaseFEM::addBilinear( // [2024 Using levelset normal]
        // -innerProduct(itfPenParam*pow(h,-1)*p_itf, q_itf) 
        -innerProduct(itfPenParam*pow(h,1)*(normal_phi.exprList()*grad(p_itf)), (normal_phi.exprList()*grad(q_itf))) 
        , Kh_itf
        );

        // [Boundary conditions]
        darcy.addBilinear(
        +innerProduct(p_itf, v*n) 
        +innerProduct(u*n, q_itf)
        , interface
        );
        darcy.addLinear(fun_exact_u,
        +innerProduct(1, q_itf*n)
        , interface
        ); 

        // [LAGRANGE MULT]
        darcy.addLagrangeMultiplier(
        +innerProduct(1, p), 0
        , Kh_i
        );

        // [Add to last diagonal entry of LHS]
        // int nb_dof = Wh.get_nb_dof()+Ph.get_nb_dof()+Ph_itf.get_nb_dof();
        // darcy.mat_[make_pair(nb_dof,nb_dof)] = area;

        // matlab::Export(darcy.mat_[0], "mat"+std::to_string(i)+"Cut.dat");
        // std::cout << "Warning: not exporting matrix" << std::endl;
        // std::getchar();
        darcy.solve("mumps");

        std::cout << "Lagrange multiplier value: " << std::endl;
        std::cout << darcy.rhs_(Wh.get_nb_dof()+Ph.get_nb_dof()+Ph_itf.get_nb_dof())<< std::endl;

        // EXTRACT SOLUTION
        int nbdof_vel = Wh.get_nb_dof();
        int nbdof_pres = Ph.get_nb_dof();
        int nbdof_pres_itf = Ph_itf.get_nb_dof();
        Rn_ data_uh = darcy.rhs_(SubArray(nbdof_vel,0));
        Rn_ data_ph = darcy.rhs_(SubArray(nbdof_pres,nbdof_vel));
        Rn_ data_ph_itf = darcy.rhs_(SubArray(nbdof_pres_itf,nbdof_pres+nbdof_vel));

        Fun_h uh(Wh, data_uh);
        Fun_h ph(Ph, data_ph);
        Fun_h ph_itf(Ph_itf, data_ph_itf);
        // std::cout << data_ph_itf << std::endl;

        auto uh_0dx = dx(uh.expr(0));
        auto uh_1dy = dy(uh.expr(1));        

        // [Post process pressure]
        R meanP = integral(Kh_i,p0.expr(),0);
        R meanPfem = integral(Kh_i,ph.expr(),0);
        // std::cout << meanP << std::endl;
        ph.v -= meanPfem/area;
        ph.v += meanP/area;

        // L2 norm vel
        R errU      = L2normCut(uh,fun_exact_u,0,2);
        R errP      = L2normCut(ph,fun_exact_p,0,1);
        R errDiv    = L2normCut (uh_0dx+uh_1dy,fun_div,Kh_i);
        R maxErrDiv = maxNormCut(uh_0dx+uh_1dy,fun_div,Kh_i);
        // R errLagr = L2normCut(ph_itf,fun_exact_p,0,1);
        R errLagr = L2normSurf(ph_itf,fun_exact_p_surf,interface,0,1);

        // [PLOTTING]
        {
        // Fun_h soluh(Wh, fun_exact_u);
        // Fun_h solph(Wh, fun_exact_p);

        // Fun_h divSolh(P2h, fun_div);
        // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

        Paraview<Mesh> writer(Kh_i, "darcyFict_"+std::to_string(i)+".vtk");
        writer.add(uh, "velocity" , 0, 2);
        writer.add(u0, "velocityExact" , 0, 2);
        writer.add(ph, "pressure" , 0, 1);
        writer.add(p0, "pressureExact" , 0, 1);
        writer.add(uh_0dx+uh_1dy, "divergence");
        // soluh.v -= uh.v; soluh.v.map(fabs);
        // writer.add(soluh, "velocityError" , 0, 2);
        writer.add(fabs((uh_0dx+uh_1dy)-fq.expr()), "divergenceError");
        // writer.add(ph_itf, "bdry pressure", 0, 1);

        // Paraview<Mesh> writerMacro(Kh, "Kh" + to_string(i) + ".vtk");
        // writerMacro.add(levelSet, "levelSet.vtk", 0, 1);
        // writerMacro.writeActiveMesh(Kh_itf, "Kh_itf"+to_string(i) + ".vtk");
        // writerMacro.writeMacroInnerEdge(macro_itf, "macro_itf_internal_edge"+to_string(i) + ".vtk");
        // writerMacro.writeMacroOutterEdge(macro_itf, "macro_itf_outer_edge"+to_string(i) + ".vtk");
        // writerMacro.writeMacroElement(macro_itf, "macro_itf"+to_string(i) + ".vtk");

        }

        hPr.push_back(h);

        pPrint.push_back(errP);
        uPrint.push_back(errU);
        divPrint.push_back(errDiv);
        maxDivPrint.push_back(maxErrDiv);
        lagrPrint.push_back(errLagr);

        if(i==0) {convpPr.push_back(0);convuPr.push_back(0);convdivPr.push_back(0);convdivPr.push_back(0);convmaxdivPr.push_back(0);convLagrPr.push_back(0);}
        else {
        convpPr.push_back( log(pPrint[i]/pPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convuPr.push_back( log(uPrint[i]/uPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convdivPr.push_back( log(divPrint[i]/divPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convmaxdivPr.push_back( log(maxDivPrint[i]/maxDivPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convLagrPr.push_back( log(lagrPrint[i]/lagrPrint[i-1])/log(hPr[i]/hPr[i-1]));
        }

        nx = 2*nx-1;
        ny = 2*ny-1;
        // nx += 1;
        // ny += 1;
        // nx = (int)round( (1+0.2*i)*nx/2 )*2; // Makes a nonuniform refinement to an EVEN integer
        // ny = (int)round( (1+0.2*i)*ny/2 )*2;
        // std::cout << nx << std::endl;
        // shift = 0.5+(i+1)*h_i/iters; // moves one grid cell over entire span
    }
    std::cout << "\n" << std::left
    << std::setw(10) << std::setfill(' ') << "h"
    << std::setw(15) << std::setfill(' ') << "err p"
    << std::setw(15) << std::setfill(' ') << "conv p"
    << std::setw(15) << std::setfill(' ') << "err u"
    << std::setw(15) << std::setfill(' ') << "conv u"
    << std::setw(15) << std::setfill(' ') << "err divu"
    // << std::setw(15) << std::setfill(' ') << "conv divu"
    << std::setw(15) << std::setfill(' ') << "err maxdivu"
    // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
    << std::setw(15) << std::setfill(' ') << "err p_itf"
    << std::setw(15) << std::setfill(' ') << "conv p_itf"    
    << "\n" << std::endl;
    for(int i=0;i<uPrint.size();++i) {
        std::cout << std::left
        << std::setw(10) << std::setfill(' ') << hPr[i]
        << std::setw(15) << std::setfill(' ') << pPrint[i]
        << std::setw(15) << std::setfill(' ') << convpPr[i]
        << std::setw(15) << std::setfill(' ') << uPrint[i]
        << std::setw(15) << std::setfill(' ') << convuPr[i]
        << std::setw(15) << std::setfill(' ') << divPrint[i]
        // << std::setw(15) << std::setfill(' ') << convdivPr[i]
        << std::setw(15) << std::setfill(' ') << maxDivPrint[i]
        // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
        << std::setw(15) << std::setfill(' ') << lagrPrint[i]
        << std::setw(15) << std::setfill(' ') << convLagrPr[i]
        << std::endl;
    }
  }

#endif

#ifdef DARCY_EXAMPLE_CIRCLE_SARA_TESTS // ex 2
  namespace Data_Darcy_Circle {
    R d_x = 1.;
    R d_y = 1.;
    R pie = M_PI;
    R shift = 0.5;

    // // [u+grad p = 0]
    // R fun_levelSet(const R2 P, const int i) {
    //   return 0.45 - sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift));
    // } // normal = 4*(x-1/2,y-1/2)
    // R fun_force(const R2 P, int compInd) {
    //   return 0;
    // }
    // R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
    //   return -8*pie*pie*sin(2*pie*P.x)*cos(2*pie*P.y);
    // }
    // R fun_exact_u(const R2 P, int compInd, int dom) {
    //   R x = P.x;
    //   R y = P.y;
    //   if (compInd==0) {
    //     return 2*pie*cos(2*pie*x)*cos(2*pie*y);
    //   } else {
    //     return -2*pie*sin(2*pie*x)*sin(2*pie*y);
    //   }
    // }
    // R fun_exact_p(const R2 P, int compInd, int dom) {
    //   return -sin(2*pie*P.x)*cos(2*pie*P.y);
    // }
    // R fun_exact_p_surf(const R2 P, int compInd) {
    //   return -sin(2*pie*P.x)*cos(2*pie*P.y);
    // }

    // [divu = 0, DOMAIN = CIRCLE WITH RADIUS 0.25 AT (0.5,0.5) ]
    R fun_levelSet(const R2 P, const int i) {
      return 0.25 - sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift));
    } // normal = 4*(x-1/2,y-1/2)
    R fun_force(const R2 P, int compInd) {
      return 0;
    }
    R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
      return 0;
    }
    R fun_exact_u(const R2 P, int compInd, int dom) {
      if (compInd==0) {
        return cos(P.x)*sinh(P.y);
      } else {
        return sin(P.x)*cosh(P.y);
      }
    }
    R fun_exact_p(const R2 P, int compInd, int dom) {
      return -sin(P.x)*sinh(P.y) - (cos(1) - 1)*(cosh(1) - 1);
    }

    // [u is linear, int_bdry u*n = pi/8]
    // R fun_force(const R2 P, int compInd) {
    //   return 0;
    // }
    // R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
    //   return 2;
    // }
    // R fun_exact_u(const R2 P, int compInd, int dom) {
    //   if (compInd==0) {
    //     return P.x-0.5;
    //   } else {
    //     return P.y-0.5;
    //   }
    // }
    // R fun_exact_p(const R2 P, int compInd, int dom) {
    //   return -0.5*(P.x*P.x-P.x + P.y*P.y-P.y);
    // }

  }
  using namespace Data_Darcy_Circle;

  int main(int argc, char** argv ) {
    typedef TestFunction<Mesh2> FunTest;
    typedef FunFEM<Mesh2> Fun_h;
    typedef Mesh2 Mesh;
    typedef ActiveMeshT2 CutMesh;
    typedef FESpace2   Space;
    typedef CutFESpaceT2 CutSpace;

    // MPIcf cfMPI(argc,argv);
    const double cpubegin = CPUtime();

    int nx =11; // 6
    int ny =11; // 6
    // Kh_init0.truncate(interface_init, -1);

    std::vector<double> hPr,uPrint,pPrint,divPrint,divPrintLoc,maxDivPrint,lagrPrint,convuPr,convpPr,convdivPr,convdivPrLoc,convmaxdivPr,convLagrPr;

    int iters = 3;
    for(int i=0; i<iters; ++i) {
        Mesh Kh(nx, ny, 0., 0., d_x+globalVariable::Epsilon, d_y+globalVariable::Epsilon);

        const R h = 1./(nx-1);
        const R invh = 1./h;

        Space V0h(Kh, DataFE<Mesh>::RT0); Space Q0h(Kh, DataFE<Mesh>::P0); // for area computation (always lowest order)
        
        // Space V2h(Kh, DataFE<Mesh>::RT2); Space Q2h(Kh, DataFE<Mesh>::P2); // for the RHS
        Lagrange2 FEvelocity2(4); Space V2h(Kh, FEvelocity2); Space Q2h(Kh, DataFE<Mesh>::P2); // for the RHS

        Space Vh(Kh, DataFE<Mesh>::RT1);
        Space Qh(Kh, DataFE<Mesh>::P1dc);
        Space Qh_itf(Kh, DataFE<Mesh>::P2);

        Space Lh(Kh, DataFE<Mesh2>::P1);
        Fun_h levelSet(Lh, fun_levelSet);
        InterfaceLevelSet<Mesh> interface(Kh, levelSet);
        Normal n;
        Tangent t;

        // MAIN MESH
        ActiveMesh<Mesh> Kh_i(Kh);       // Kh_i.info();
        Kh_i.truncate(interface, -1); // [-1 removes the negative part]

        CutSpace W0h(Kh_i, V0h); CutSpace P0h(Kh_i, Q0h);
        CutSpace W2h(Kh_i, V2h); CutSpace P2h(Kh_i, Q2h);

        CutSpace Wh(Kh_i, Vh);// Wh.info();
        CutSpace Ph(Kh_i, Qh);

        // MacroElement<Mesh> macro(Kh_i, 1);
        
        // SURFACE MESH
        // ActiveMesh<Mesh> Kh_itf(Kh);
        // Kh_itf.createSurfaceMesh(interface);
        // CutSpace Ph_itf(Kh_itf, Qh_itf);
        // MacroElementSurface<Mesh> macro_itf(Kh_itf, interface, 0.3); // 1

        // Test functions
        FunTest p(Ph,1), q(Ph,1), u(Wh,2), v(Wh,2);
        // FunTest p_itf(Ph_itf,1), q_itf(Ph_itf,1);

        // Fun_h fv(Wh, fun_force);
        Fun_h fq_(P2h, fun_div);
        Fun_h p0(P2h, fun_exact_p);
        Fun_h u0(W2h, fun_exact_u);

        // Lerenfeld interpolation
        CutFEM<Mesh2> cutfem_interp(Ph);
        cutfem_interp.addBilinear(
        +innerProduct(p, q)
        , Kh_i
        ); 
        cutfem_interp.addPatchStabilization( // [h^(2k) h^(2k)]
        +innerProduct(jump(p), jump(q))
        , Kh_i
        );
        cutfem_interp.addLinear(
        +innerProduct(fq_.expr(), q)
        , Kh_i
        ); 
        cutfem_interp.solve("umfpack");
        Rn_ data_div = cutfem_interp.rhs_(SubArray(Ph.get_nb_dof(),0));
        Fun_h fq(Ph, data_div);

        // [Area]
        CutFEM<Mesh2> area_comp(P0h); FunTest q0(P0h,1);
        area_comp.addLinear(
        +innerProduct(1, q0)
        , Kh_i
        ); 
        R area = area_comp.rhs_.sum();

        // Create CutFEM object
        CutFEM<Mesh2> darcy(Wh); darcy.add(Ph); //darcy.add(Ph_itf);
        // [ASSEMBLY]
        darcy.addBilinear(
        +innerProduct(u, v)
        , Kh_i
        );
        darcy.addBilinear(
        -innerProduct(p, div(v))
        , Kh_i
        );
        darcy.addBilinear(
        +innerProduct(div(u), q)
        , Kh_i, INTEGRAL_EXTENSION, 1
        );
        darcy.addLinear(
        // innerProduct(fv.expression(2), v)
        +innerProduct(fq.exprList(), q)
        , Kh_i, INTEGRAL_EXTENSION, 1
        );


        // [GHOST PENALTY]
        double uPenParam = 1e0; // 1e0 
        double pPenParam = 1e0; // 1e0
        double itfPenParam = 1e0; // 1e-2

        FunTest grad2un = grad(grad(u)*n)*n;
        // FunTest grad2pn = grad(grad(p)*n)*n;
        // FunTest grad2divun = grad(grad(div(u))*n)*n;
        // FunTest grad2pitfn = grad(grad(p_itf)*n)*n;
        darcy.addPatchStabilization( // [h^(2k) h^(2k)]
        +innerProduct(uPenParam*jump(u), jump(v)) 
        -innerProduct(pPenParam*jump(p), jump(div(v)))
        // +innerProduct(pPenParam*jump(div(u)), jump(q))
        , Kh_i
        // , macro
        );
        // darcy.addFaceStabilization(
        // // -innerProduct(itfPenParam*pow(h,-1)*jump(p_itf), jump(q_itf))
        // -innerProduct(itfPenParam*pow(h,3)*jump(grad(p_itf)*n), jump(grad(q_itf)*n)) 
        // -innerProduct(itfPenParam*pow(h,5)*jump(grad2pitfn), jump(grad2pitfn))
        // , Kh_itf
        // // , macro_itf
        // );
        // darcy.addBilinear(
        //   // -innerProduct(itfPenParam*pow(h,0)*jump(p_itf), jump(q_itf))
        //   -innerProduct(itfPenParam*pow(h,3)*jump(grad(p_itf)*n), jump(grad(q_itf)*n)) 
        //   -innerProduct(itfPenParam*pow(h,5)*jump(grad2pitfn), jump(grad2pitfn))
        // , interface
        // );
        // darcy.BaseFEM::addBilinear(
        //   -innerProduct(itfPenParam*pow(h,3)*grad(p_itf)*n, grad(q_itf)*n) // scaling like the interface stab!
        // , Kh_itf
        // );

        // [Boundary conditions]
        // darcy.addBilinear(
        //   +innerProduct(p_itf, v*n) 
        //   +innerProduct(u*n, q_itf)
        //   , interface
        // );
        // darcy.addLinear(fun_exact_u,
        //   +innerProduct(1, q_itf*n)
        //   , interface
        // ); 
        // R pp = 1e2;
        // darcy.addBilinear(
        //   +innerProduct(p, v*n) 
        //   +innerProduct(u*n, pp * 1./h *v*n)
        //   , interface
        // );
        // darcy.addLinear(
        //   +innerProduct(u0*n, pp * 1./h *v*n)
        //   , interface
        // ); 
        darcy.addLinear(
          -innerProduct(p0.expr(), v*n)
          , interface
        ); 

        // [LAGRANGE MULT]
        // darcy.addLagrangeMultiplier(
        // +innerProduct(1, p), 0
        // , Kh_i
        // );

        // [Add to last diagonal entry of LHS]
        // int nb_dof = Wh.get_nb_dof()+Ph.get_nb_dof()+Ph_itf.get_nb_dof();
        // darcy.mat_[make_pair(nb_dof,nb_dof)] = area;

        matlab::Export(darcy.mat_[0], "mat"+std::to_string(i)+"Cut.dat");
        darcy.solve("umfpack");

        // std::cout << "Lagrange multiplier value: " << std::endl;
        // std::cout << darcy.rhs_(Wh.get_nb_dof()+Ph.get_nb_dof())<< std::endl;
        // std::cout << darcy.rhs_(Wh.get_nb_dof()+Ph.get_nb_dof()+Ph_itf.get_nb_dof())<< std::endl;

        // EXTRACT SOLUTION
        int nbdof_vel = Wh.get_nb_dof();
        int nbdof_pres = Ph.get_nb_dof();
        // int nbdof_pres_itf = Ph_itf.get_nb_dof();
        Rn_ data_uh = darcy.rhs_(SubArray(nbdof_vel,0));
        Rn_ data_ph = darcy.rhs_(SubArray(nbdof_pres,nbdof_vel));
        // Rn_ data_ph_itf = darcy.rhs_(SubArray(nbdof_pres_itf,nbdof_pres+nbdof_vel));

        Fun_h uh(Wh, data_uh);
        Fun_h ph(Ph, data_ph);
        // Fun_h ph_itf(Ph_itf, data_ph_itf);
        // std::cout << data_ph_itf << std::endl;

        auto uh_0dx = dx(uh.expr(0));
        auto uh_1dy = dy(uh.expr(1));        

        // [Post process pressure]
        R meanP = integral(Kh_i,p0.expr(),0);
        R meanPfem = integral(Kh_i,ph.expr(),0);
        // std::cout << meanP << std::endl;
        ph.v -= meanPfem/area;
        ph.v += meanP/area;

        // L2 norm vel
        R errU      = L2normCut(uh,fun_exact_u,0,2);
        R errP      = L2normCut(ph,fun_exact_p,0,1);
        R errDiv    = L2normCut (uh_0dx+uh_1dy,fun_div,Kh_i);
        R maxErrDiv = maxNormCut(uh_0dx+uh_1dy,fun_div,Kh_i);
        // R errLagr = L2normCut(ph_itf,fun_exact_p,0,1);
        R errLagr = 0; //L2normSurf(ph_itf,fun_exact_p_surf,interface,0,1);

        // [PLOTTING]
        {
        // Fun_h soluh(Wh, fun_exact_u);
        // Fun_h solph(Wh, fun_exact_p);

        // Fun_h divSolh(P2h, fun_div);
        // ExpressionFunFEM<Mesh> femDiv(divSolh, 0, op_id);

        Paraview<Mesh> writer(Kh_i, "darcyFict_"+std::to_string(i)+".vtk");
        writer.add(uh, "velocity" , 0, 2);
        writer.add(u0, "velocityExact" , 0, 2);
        writer.add(ph, "pressure" , 0, 1);
        writer.add(p0, "pressureExact" , 0, 1);
        writer.add(uh_0dx+uh_1dy, "divergence");
        // soluh.v -= uh.v; soluh.v.map(fabs);
        // writer.add(soluh, "velocityError" , 0, 2);
        writer.add(fabs((uh_0dx+uh_1dy)-fq.expr()), "divergenceError");
        // writer.add(ph_itf, "bdry pressure", 0, 1);

        // Paraview<Mesh> writerMacro(Kh, "Kh" + to_string(i) + ".vtk");
        // writerMacro.add(levelSet, "levelSet.vtk", 0, 1);
        // writerMacro.writeActiveMesh(Kh_itf, "Kh_itf"+to_string(i) + ".vtk");
        // writerMacro.writeMacroInnerEdge(macro_itf, "macro_itf_internal_edge"+to_string(i) + ".vtk");
        // writerMacro.writeMacroOutterEdge(macro_itf, "macro_itf_outer_edge"+to_string(i) + ".vtk");
        // writerMacro.writeMacroElement(macro_itf, "macro_itf"+to_string(i) + ".vtk");

        }

        hPr.push_back(h);

        pPrint.push_back(errP);
        uPrint.push_back(errU);
        divPrint.push_back(errDiv);
        maxDivPrint.push_back(maxErrDiv);
        lagrPrint.push_back(errLagr);

        if(i==0) {convpPr.push_back(0);convuPr.push_back(0);convdivPr.push_back(0);convdivPr.push_back(0);convmaxdivPr.push_back(0);convLagrPr.push_back(0);}
        else {
        convpPr.push_back( log(pPrint[i]/pPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convuPr.push_back( log(uPrint[i]/uPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convdivPr.push_back( log(divPrint[i]/divPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convmaxdivPr.push_back( log(maxDivPrint[i]/maxDivPrint[i-1])/log(hPr[i]/hPr[i-1]));
        convLagrPr.push_back( log(lagrPrint[i]/lagrPrint[i-1])/log(hPr[i]/hPr[i-1]));
        }

        nx = 2*nx-1;
        ny = 2*ny-1;
        // nx += 1;
        // ny += 1;
        // nx = (int)round( (1+0.2*i)*nx/2 )*2; // Makes a nonuniform refinement to an EVEN integer
        // ny = (int)round( (1+0.2*i)*ny/2 )*2;
        // std::cout << nx << std::endl;
        // shift = 0.5+(i+1)*h_i/iters; // moves one grid cell over entire span
    }
    std::cout << "\n" << std::left
    << std::setw(10) << std::setfill(' ') << "h"
    << std::setw(15) << std::setfill(' ') << "err p"
    << std::setw(15) << std::setfill(' ') << "conv p"
    << std::setw(15) << std::setfill(' ') << "err u"
    << std::setw(15) << std::setfill(' ') << "conv u"
    << std::setw(15) << std::setfill(' ') << "err divu"
    // << std::setw(15) << std::setfill(' ') << "conv divu"
    << std::setw(15) << std::setfill(' ') << "err maxdivu"
    // << std::setw(15) << std::setfill(' ') << "conv maxdivu"
    << std::setw(15) << std::setfill(' ') << "err p_itf"
    << std::setw(15) << std::setfill(' ') << "conv p_itf"    
    << "\n" << std::endl;
    for(int i=0;i<uPrint.size();++i) {
        std::cout << std::left
        << std::setw(10) << std::setfill(' ') << hPr[i]
        << std::setw(15) << std::setfill(' ') << pPrint[i]
        << std::setw(15) << std::setfill(' ') << convpPr[i]
        << std::setw(15) << std::setfill(' ') << uPrint[i]
        << std::setw(15) << std::setfill(' ') << convuPr[i]
        << std::setw(15) << std::setfill(' ') << divPrint[i]
        // << std::setw(15) << std::setfill(' ') << convdivPr[i]
        << std::setw(15) << std::setfill(' ') << maxDivPrint[i]
        // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
        << std::setw(15) << std::setfill(' ') << lagrPrint[i]
        << std::setw(15) << std::setfill(' ') << convLagrPr[i]
        << std::endl;
    }
  }

#endif

#ifdef DARCY_EXAMPLE_TWO_BC_ANNULUS_PLUS_INTERFACE

  namespace Data_Darcy_Two_Unfitted {
    R d_x = 1.;
    R d_y = 1.;
    R shift = 0.5;
    R inRad  = 0.15;
    R interfaceRad = 0.3; // 0.350001
    R outRad = 0.45; // 0.4901

    R pie = M_PI;//3.14159265359;

    R fun_levelSet_in(const R2 P, const int i) {
      return sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift)) - inRad;
    }
    R fun_levelSet(const R2 P, const int i) {
      return sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift)) - interfaceRad;
    }
    R fun_levelSet_out(const R2 P, const int i) {
      return outRad - sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift));
    }
    R fun_test(const R2 P,const int i, int d) {return d;}

    // [Eriks Scotti example]
    R rad2 = interfaceRad*interfaceRad;
    R mu_G = 2*interfaceRad/(4*cos(rad2)+3); // xi0*mu_G = 1/8*2/3*1/4
    R phat = (19*rad2+12*sin(rad2)+8*sin(2*rad2)+24*rad2*cos(rad2))/(4*rad2*(4*cos(rad2)+3));
    R xi0 = 1./8;

    R fun_force(const R2 P, int compInd) {
      return 0;
    }
    R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
      R r2 = (P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift);
      if (dom==0) // r2>radius2
        return 2./rad2*(2*r2*sin(r2)-2*cos(r2)-1);
      else
        return 4./rad2*(r2*sin(r2)-cos(r2)-1);
    }
    R fun_exact_u(const R2 P, int compInd, int dom) {
      DA<R,2> X(P.x,0), Y(P.y,1);
      DA<R,2> r2 = (X-shift)*(X-shift) + (Y-shift)*(Y-shift);
      R cst = (dom==0)*3./2;
      R mul = (dom==0)*2 + (dom==1)*1;
      DA<R, 2> val = r2/(mul*rad2) + cst + sin(r2)/rad2;
      return -val.d[compInd];
    }
    R fun_exact_p(const R2 P, int compInd, int dom) {
      DA<R,2> X(P.x,0), Y(P.y,1);
      DA<R,2> r2 = (X-shift)*(X-shift) + (Y-shift)*(Y-shift);
      R cst = (dom==0)*3./2;
      R mul = (dom==0)*2 + (dom==1)*1;
      DA<R, 2> val = r2/(mul*rad2) + cst + sin(r2)/rad2;
      return val.val;
    }
    R fun_exact_p_surf(const R2 P, int compInd) {
      DA<R,2> X(P.x,0), Y(P.y,1);
      DA<R,2> r2 = (X-shift)*(X-shift) + (Y-shift)*(Y-shift);
      R cst = 3./2;
      R mul = 2;
      DA<R, 2> val = r2/(mul*rad2) + cst + sin(r2)/rad2;
      return val.val;
    }

  }
  using namespace Data_Darcy_Two_Unfitted;

  int main(int argc, char** argv ) {
    typedef TestFunction<Mesh2> FunTest;
    typedef FunFEM<Mesh2> Fun_h;
    typedef Mesh2 Mesh;
    typedef ActiveMeshT2 CutMesh;
    typedef FESpace2   Space;
    typedef CutFESpaceT2 CutSpace;

    // MPIcf cfMPI(argc,argv);
    const double cpubegin = CPUtime();

    int nx = 11; // 6
    int ny = 11; // 6

    std::vector<double> uPrint,pPrint,divPrint,divPrintLoc,maxDivPrint,h,lagrPrint,convuPr,convpPr,convdivPr,convdivPrLoc,convmaxdivPr,convLagrPr;

    int iters = 5;

    for(int i=0;i<iters;++i) {
      std::cout << "Iteration " << i << std::endl;
      Mesh Kh(nx, ny, 0., 0., d_x+globalVariable::Epsilon, d_y+globalVariable::Epsilon);
      const R h_i = 1./(nx-1);
      const R invh = 1./h_i;

      Space Lh(Kh, DataFE<Mesh2>::P1);

      Fun_h levelSet_out(Lh, fun_levelSet_out);
      InterfaceLevelSet<Mesh> boundary_out(Kh, levelSet_out);

      Fun_h levelSet(Lh, fun_levelSet);
      InterfaceLevelSet<Mesh> interface(Kh, levelSet);

      Fun_h levelSet_in(Lh, fun_levelSet_in);
      InterfaceLevelSet<Mesh> boundary_in(Kh, levelSet_in);

      Space Vh(Kh, DataFE<Mesh>::RT0);
      Space Qh(Kh, DataFE<Mesh>::P0);
      Space Qh_itf(Kh, DataFE<Mesh>::P1dc);

      // Space V2h(Kh, DataFE<Mesh>::RT2);
      Lagrange2 FEvelocity2(4); Space V2h(Kh, FEvelocity2);
      Space Q2h(Kh, DataFE<Mesh>::P4);

      // Cut mesh
      ActiveMesh<Mesh> Kh_i(Kh);
      Kh_i.truncate(boundary_in, -1);
      Kh_i.truncate(boundary_out, -1);
      Kh_i.add(interface, -1);

      // Surface mesh
      ActiveMesh<Mesh> Kh_itf(Kh);
      Kh_itf.createSurfaceMesh(boundary_out);
      CutSpace Ph_itf(Kh_itf, Qh_itf);


      CutSpace Wh(Kh_i, Vh);
      CutSpace Ph(Kh_i, Qh);
      
      CutSpace W2h(Kh_i, V2h);
      CutSpace P2h(Kh_i, Q2h);

      // MacroElement<Mesh> macro(Kh_i, 1);


      CutFEM<Mesh2> darcy(Wh); darcy.add(Ph); darcy.add(Ph_itf);

      // We define fh on the cutSpace
      Fun_h fq(P2h, fun_div);
      Fun_h p0(P2h, fun_exact_p);
      Fun_h u0(W2h, fun_exact_u);

      Normal n;
      Tangent t;
      FunTest p(Ph,1), q(Ph,1), u(Wh,2), v(Wh,2);
      FunTest p_itf(Ph_itf,1), q_itf(Ph_itf,1);

      // double penParam = 1e-2; // 7e-3


      // [ASSEMBLY]
      darcy.addBilinear(
        innerProduct(u, v)
        -innerProduct(p, div(v))
        +innerProduct(div(u), q)
        , Kh_i
      );
      darcy.addLinear(
        innerProduct(fq.expr(), q) 
        , Kh_i
      );

      darcy.addBilinear(
        innerProduct(mu_G*average(u*n), average(v*n))
        +innerProduct(xi0*jump(u*n), mu_G*jump(v*n)) // b(p,v)-b(q,u) bdry terms
      , interface);
      darcy.addLinear(
        -innerProduct(phat, jump(v*n))
      , interface);

      darcy.addBilinear(
        innerProduct(p_itf, v*n)
        +innerProduct(u*n, q_itf)
        ,boundary_out
      );
      darcy.addLinear(
        +innerProduct(u0*n, q_itf)
        ,boundary_out
      );

      darcy.addLinear(
        -innerProduct(p0.expr(), v*n) // Only on Gamma_N (pressure)
        , boundary_in
      );

      double uPenParam = 1e0; // 1e-2
      double pPenParam = 1e0; // 1e-2
      double itfPenParam = 1e0; // 1e-2
      FunTest grad2un = grad(grad(u)*n)*n;
      // FunTest grad2pn = grad(grad(p)*n)*n;
      FunTest grad2pitfn = grad(grad(p_itf)*n)*n;
      darcy.addPatchStabilization( // [h^(2k) h^(2k)]
        innerProduct(uPenParam*jump(u), jump(v)) 
        -innerProduct(pPenParam*jump(p), jump(div(v)))
        +innerProduct(pPenParam*jump(div(u)), jump(q))
      , Kh_i
      );
      darcy.addFaceStabilization(
        -innerProduct(itfPenParam*pow(h_i,-1)*jump(p_itf), jump(q_itf))
        -innerProduct(itfPenParam*pow(h_i,1)*jump(grad(p_itf)), jump(grad(q_itf)))
        -innerProduct(itfPenParam*pow(h_i,3)*jump(grad(grad(p_itf))), jump(grad(grad(p_itf))))
        , Kh_itf
        // , macro_itf
      );
      darcy.addBilinear(
        // -innerProduct(itfPenParam*pow(h_i,-1)*jump(p_itf), jump(q_itf))
        -innerProduct(itfPenParam*pow(h_i,3)*jump(grad(p_itf)*n), jump(grad(q_itf)*n)) 
        -innerProduct(itfPenParam*pow(h_i,5)*jump(grad2pitfn), jump(grad2pitfn))
        , boundary_out
      );
      // darcy.BaseFEM::addBilinear(
      //   -innerProduct(itfPenParam*pow(h_i,1)*grad(p_itf)*n, grad(q_itf)*n) // scaling like the interface stab!
      // , Kh_itf
      // );
      
      matlab::Export(darcy.mat_[0], "mat"+std::to_string(i)+"Cut.dat");
      // return 0;
      darcy.solve("umfpack");

      // EXTRACT SOLUTION
      int nb_vel_dof = Wh.get_nb_dof();
      int nb_pres_dof = Ph.get_nb_dof();
      Rn_ data_uh = darcy.rhs_(SubArray(nb_vel_dof,0));
      Rn_ data_ph = darcy.rhs_(SubArray(nb_pres_dof,nb_vel_dof));
      Rn_ data_ph_itf = darcy.rhs_(SubArray(Ph_itf.get_nb_dof(),nb_pres_dof+nb_vel_dof));
      Fun_h uh(Wh, data_uh);
      Fun_h ph(Ph, data_ph);
      Fun_h ph_itf(Ph_itf, data_ph_itf);
      
      auto uh_0dx = dx(uh.expr(0));
      auto uh_1dy = dy(uh.expr(1));

      // L2 norm vel
      R errU      = L2normCut(uh,fun_exact_u,0,2);
      R errP      = L2normCut(ph,fun_exact_p,0,1);
      R errDiv    = L2normCut (uh_0dx+uh_1dy,fun_div,Kh_i);
      R maxErrDiv = maxNormCut(uh_0dx+uh_1dy,fun_div,Kh_i);
      R errLagr   = L2normSurf(ph_itf,fun_exact_p_surf,boundary_out,0,1);

      // [PLOTTING]
      {
        // // Fun_h solh(Wh, fun_exact);
        // // solh.v -= uh.v;
        // // solh.v.map(fabs);
        // Fun_h f_domain(Ph, fun_test);
        Paraview<Mesh> writer(Kh_i, "darcyTwoBoundaries_"+std::to_string(i)+".vtk");
        writer.add(uh, "velocity" , 0, 2);
        writer.add(u0, "velocity" , 0, 2);
        writer.add(ph, "pressure" , 0, 1);
        writer.add(p0, "pressure" , 0, 1);
        writer.add(uh_0dx+uh_1dy, "divergence");
        // // writer.add(solh, "velocityError" , 0, 2);
        writer.add(fabs((uh_0dx+uh_1dy)-fq.expr()), "divergenceError");
        // writer.add(f_domain, "domain", 0, 1);
        writer.add(ph_itf, "Lagrange multiplier" , 0, 1);
      }


      h.push_back(h_i);
      pPrint.push_back(errP);
      uPrint.push_back(errU);
      divPrint.push_back(errDiv);
      maxDivPrint.push_back(maxErrDiv);
      lagrPrint.push_back(errLagr);

      if(i==0) {convpPr.push_back(0);convuPr.push_back(0);convdivPr.push_back(0);convdivPrLoc.push_back(0);convmaxdivPr.push_back(0);convLagrPr.push_back(0);}
      else {
        convpPr.push_back( log(pPrint[i]/pPrint[i-1])/log(h[i]/h[i-1]));
        convuPr.push_back( log(uPrint[i]/uPrint[i-1])/log(h[i]/h[i-1]));
        convdivPr.push_back( log(divPrint[i]/divPrint[i-1])/log(h[i]/h[i-1]));
        convmaxdivPr.push_back( log(maxDivPrint[i]/maxDivPrint[i-1])/log(h[i]/h[i-1]));
        convLagrPr.push_back( log(lagrPrint[i]/lagrPrint[i-1])/log(h[i]/h[i-1]));
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
    << std::setw(15) << std::setfill(' ') << "err lagr"
    << std::setw(15) << std::setfill(' ') << "conv lagr"
    << "\n" << std::endl;
    for(int i=0;i<uPrint.size();++i) {
      std::cout << std::left
      << std::setw(10) << std::setfill(' ') << h[i]
      << std::setw(15) << std::setfill(' ') << pPrint[i]
      << std::setw(15) << std::setfill(' ') << convpPr[i]
      << std::setw(15) << std::setfill(' ') << uPrint[i]
      << std::setw(15) << std::setfill(' ') << convuPr[i]
      << std::setw(15) << std::setfill(' ') << divPrint[i]
      // << std::setw(15) << std::setfill(' ') << convdivPr[i]
      << std::setw(15) << std::setfill(' ') << maxDivPrint[i]
      // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
      << std::setw(15) << std::setfill(' ') << lagrPrint[i]
      << std::setw(15) << std::setfill(' ') << convLagrPr[i]
      << std::endl;
    }
  }

#endif

#ifdef DARCY_2FIELD_EXAMPLE_TWO_BC_ANNULUS_PLUS_INTERFACE

  namespace Data_Darcy_Two_Unfitted {
    R d_x = 1.;
    R d_y = 1.;
    R shift = 0.5;
    R inRad  = 0.15;
    R interfaceRad = 0.3; // 0.350001
    R outRad = 0.45; // 0.4901

    R pie = M_PI;//3.14159265359;

    R fun_levelSet_in(const R2 P, const int i) {
      return sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift)) - inRad;
    }
    R fun_levelSet(const R2 P, const int i) {
      return sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift)) - interfaceRad;
    }
    R fun_levelSet_out(const R2 P, const int i) {
      return outRad - sqrt((P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift));
    }
    R fun_test(const R2 P,const int i, int d) {return d;}

    // [Eriks Scotti example]
    R rad2 = interfaceRad*interfaceRad;
    R mu_G = 2*interfaceRad/(4*cos(rad2)+3); // xi0*mu_G = 1/8*2/3*1/4
    R phat = (19*rad2+12*sin(rad2)+8*sin(2*rad2)+24*rad2*cos(rad2))/(4*rad2*(4*cos(rad2)+3));
    R xi0 = 1./8;

    R fun_force(const R2 P, int compInd) {
      return 0;
    }
    R fun_div(const R2 P, int compInd, int dom) {// is also exact divergence
      R r2 = (P.x-shift)*(P.x-shift) + (P.y-shift)*(P.y-shift);
      if (dom==0) // r2>radius2
        return 2./rad2*(2*r2*sin(r2)-2*cos(r2)-1);
      else
        return 4./rad2*(r2*sin(r2)-cos(r2)-1);
    }
    R fun_exact_u(const R2 P, int compInd, int dom) {
      DA<R,2> X(P.x,0), Y(P.y,1);
      DA<R,2> r2 = (X-shift)*(X-shift) + (Y-shift)*(Y-shift);
      R cst = (dom==0)*3./2;
      R mul = (dom==0)*2 + (dom==1)*1;
      DA<R, 2> val = r2/(mul*rad2) + cst + sin(r2)/rad2;
      return -val.d[compInd];
    }
    R fun_exact_p(const R2 P, int compInd, int dom) {
      DA<R,2> X(P.x,0), Y(P.y,1);
      DA<R,2> r2 = (X-shift)*(X-shift) + (Y-shift)*(Y-shift);
      R cst = (dom==0)*3./2;
      R mul = (dom==0)*2 + (dom==1)*1;
      DA<R, 2> val = r2/(mul*rad2) + cst + sin(r2)/rad2;
      return val.val;
    }
    R fun_exact_p_surf(const R2 P, int compInd) {
      DA<R,2> X(P.x,0), Y(P.y,1);
      DA<R,2> r2 = (X-shift)*(X-shift) + (Y-shift)*(Y-shift);
      R cst = 3./2;
      R mul = 2;
      DA<R, 2> val = r2/(mul*rad2) + cst + sin(r2)/rad2;
      return val.val;
    }

  }
  using namespace Data_Darcy_Two_Unfitted;

  int main(int argc, char** argv ) {
    typedef TestFunction<Mesh2> FunTest;
    typedef FunFEM<Mesh2> Fun_h;
    typedef Mesh2 Mesh;
    typedef ActiveMeshT2 CutMesh;
    typedef FESpace2   Space;
    typedef CutFESpaceT2 CutSpace;

    // MPIcf cfMPI(argc,argv);
    const double cpubegin = CPUtime();

    int nx = 11; // 6
    int ny = 11; // 6

    std::vector<double> uPrint,pPrint,divPrint,divPrintLoc,maxDivPrint,h,lagrPrint,convuPr,convpPr,convdivPr,convdivPrLoc,convmaxdivPr,convLagrPr;

    int iters = 5;

    for(int i=0;i<iters;++i) {
      std::cout << "Iteration " << i << std::endl;
      Mesh Kh(nx, ny, 0., 0., d_x+globalVariable::Epsilon, d_y+globalVariable::Epsilon);
      const R h_i = 1./(nx-1);
      const R invh = 1./h_i;

      Space Lh(Kh, DataFE<Mesh2>::P1);

      Fun_h levelSet_out(Lh, fun_levelSet_out);
      InterfaceLevelSet<Mesh> boundary_out(Kh, levelSet_out);

      Fun_h levelSet(Lh, fun_levelSet);
      InterfaceLevelSet<Mesh> interface(Kh, levelSet);

      Fun_h levelSet_in(Lh, fun_levelSet_in);
      InterfaceLevelSet<Mesh> boundary_in(Kh, levelSet_in);

      Space Vh(Kh, DataFE<Mesh>::RT1);
      Space Qh(Kh, DataFE<Mesh>::P1dc);

      // Space V2h(Kh, DataFE<Mesh>::RT2);
      Lagrange2 FEvelocity2(4); Space V2h(Kh, FEvelocity2);
      Space Q2h(Kh, DataFE<Mesh>::P4);

      // Cut mesh
      ActiveMesh<Mesh> Kh_i(Kh);
      Kh_i.truncate(boundary_in, -1);
      Kh_i.truncate(boundary_out, -1);
      Kh_i.add(interface, -1);

      CutSpace Wh(Kh_i, Vh);
      CutSpace Ph(Kh_i, Qh);
      
      CutSpace W2h(Kh_i, V2h);
      CutSpace P2h(Kh_i, Q2h);

      // MacroElement<Mesh> macro(Kh_i, 1);

      CutFEM<Mesh2> darcy(Wh); darcy.add(Ph);;

      // We define fh on the cutSpace
      Fun_h fq(P2h, fun_div);
      Fun_h p0(P2h, fun_exact_p);
      Fun_h u0(W2h, fun_exact_u);

      Normal n;
      Tangent t;
      FunTest p(Ph,1), q(Ph,1), u(Wh,2), v(Wh,2);


      // [ASSEMBLY]
      darcy.addBilinear(
        innerProduct(u, v)
        -innerProduct(p, div(v))
        +innerProduct(div(u), q)
        , Kh_i
      );
      darcy.addLinear(
        innerProduct(fq.expr(), q) 
        , Kh_i
      );

      darcy.addBilinear(
        innerProduct(mu_G*average(u*n), average(v*n))
        +innerProduct(xi0*jump(u*n), mu_G*jump(v*n)) // b(p,v)-b(q,u) bdry terms
      , interface);
      darcy.addLinear(
        -innerProduct(phat, jump(v*n))
      , interface);

      R pp = 1e2;
      darcy.addBilinear(
        innerProduct(p, v*n)
        // +innerProduct(u*n, q)
        +innerProduct(u*n, pp*invh * v*n)
        // +innerProduct(u, pp*invh * v)
        ,boundary_out
      );
      darcy.addLinear(
        // +innerProduct(u0*n, q)
        +innerProduct(u0*n, pp*invh * v*n)
        // +innerProduct(u0.exprList(), pp*invh * v)
        ,boundary_out
      );

      darcy.addLinear(
        -innerProduct(p0.expr(), v*n) // Only on Gamma_N (pressure)
        , boundary_in
      );

      double uPenParam = 1e0; // 1e-2
      double pPenParam = 1e0; // 1e-2
      darcy.addPatchStabilization( // [h^(2k) h^(2k)]
        innerProduct(uPenParam*jump(u), jump(v)) 
        -innerProduct(pPenParam*jump(p), jump(div(v)))
        +innerProduct(pPenParam*jump(div(u)), jump(q))
      , Kh_i
      // , macro
      );
      // darcy.addFaceStabilization( // [h^(2k) h^(2k)]
      //   innerProduct(uPenParam*h_i*jump(u), jump(v)) 
      //   +innerProduct(uPenParam*pow(h_i,3)*jump(grad(u)), jump(grad(v))) 
      //   -innerProduct(pPenParam*h_i*jump(p), jump(div(v)))
      //   +innerProduct(pPenParam*h_i*jump(div(u)), jump(q))
      // , Kh_i
      // // , macro
      // );
      
      matlab::Export(darcy.mat_[0], "mat"+std::to_string(i)+"Cut.dat");
      // return 0;
      darcy.solve("umfpack");

      // EXTRACT SOLUTION
      int nb_vel_dof = Wh.get_nb_dof();
      int nb_pres_dof = Ph.get_nb_dof();
      Rn_ data_uh = darcy.rhs_(SubArray(nb_vel_dof,0));
      Rn_ data_ph = darcy.rhs_(SubArray(nb_pres_dof,nb_vel_dof));
      Fun_h uh(Wh, data_uh);
      Fun_h ph(Ph, data_ph);
      
      auto uh_0dx = dx(uh.expr(0));
      auto uh_1dy = dy(uh.expr(1));

      // L2 norm vel
      R errU      = L2normCut(uh,fun_exact_u,0,2);
      R errP      = L2normCut(ph,fun_exact_p,0,1);
      R errDiv    = L2normCut (uh_0dx+uh_1dy,fun_div,Kh_i);
      R maxErrDiv = maxNormCut(uh_0dx+uh_1dy,fun_div,Kh_i);
      R errLagr   = 0;

      // [PLOTTING]
      {
        // // Fun_h solh(Wh, fun_exact);
        // // solh.v -= uh.v;
        // // solh.v.map(fabs);
        // Fun_h f_domain(Ph, fun_test);
        Paraview<Mesh> writer(Kh_i, "darcyTwoBoundaries_"+std::to_string(i)+".vtk");
        writer.add(uh, "velocity" , 0, 2);
        writer.add(u0, "velocity" , 0, 2);
        writer.add(ph, "pressure" , 0, 1);
        writer.add(p0, "pressure" , 0, 1);
        writer.add(uh_0dx+uh_1dy, "divergence");
        // // writer.add(solh, "velocityError" , 0, 2);
        writer.add(fabs((uh_0dx+uh_1dy)-fq.expr()), "divergenceError");
        // writer.add(f_domain, "domain", 0, 1);
      }


      h.push_back(h_i);
      pPrint.push_back(errP);
      uPrint.push_back(errU);
      divPrint.push_back(errDiv);
      maxDivPrint.push_back(maxErrDiv);
      lagrPrint.push_back(errLagr);

      if(i==0) {convpPr.push_back(0);convuPr.push_back(0);convdivPr.push_back(0);convdivPrLoc.push_back(0);convmaxdivPr.push_back(0);convLagrPr.push_back(0);}
      else {
        convpPr.push_back( log(pPrint[i]/pPrint[i-1])/log(h[i]/h[i-1]));
        convuPr.push_back( log(uPrint[i]/uPrint[i-1])/log(h[i]/h[i-1]));
        convdivPr.push_back( log(divPrint[i]/divPrint[i-1])/log(h[i]/h[i-1]));
        convmaxdivPr.push_back( log(maxDivPrint[i]/maxDivPrint[i-1])/log(h[i]/h[i-1]));
        convLagrPr.push_back( log(lagrPrint[i]/lagrPrint[i-1])/log(h[i]/h[i-1]));
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
    << std::setw(15) << std::setfill(' ') << "err lagr"
    << std::setw(15) << std::setfill(' ') << "conv lagr"
    << "\n" << std::endl;
    for(int i=0;i<uPrint.size();++i) {
      std::cout << std::left
      << std::setw(10) << std::setfill(' ') << h[i]
      << std::setw(15) << std::setfill(' ') << pPrint[i]
      << std::setw(15) << std::setfill(' ') << convpPr[i]
      << std::setw(15) << std::setfill(' ') << uPrint[i]
      << std::setw(15) << std::setfill(' ') << convuPr[i]
      << std::setw(15) << std::setfill(' ') << divPrint[i]
      // << std::setw(15) << std::setfill(' ') << convdivPr[i]
      << std::setw(15) << std::setfill(' ') << maxDivPrint[i]
      // << std::setw(15) << std::setfill(' ') << convmaxdivPr[i]
      << std::setw(15) << std::setfill(' ') << lagrPrint[i]
      << std::setw(15) << std::setfill(' ') << convLagrPr[i]
      << std::endl;
    }
  }

#endif