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

}
using namespace Data_Darcy_Square;

int main(int argc, char** argv ) {
typedef TestFunction<Mesh2> FunTest;
typedef FunFEM<Mesh2> Fun_h;
typedef Mesh2 Mesh;
typedef ActiveMeshT2 CutMesh;
typedef FESpace2   Space;
typedef CutFESpaceT2 CutSpace;

MPIcf cfMPI(argc,argv);
const double cpubegin = CPUtime();

int nx =10; // 6
int ny =10; // 6
// Kh_init0.truncate(interface_init, -1);

std::vector<double> hPr,uPrint,pPrint,divPrint,divPrintLoc,maxDivPrint,convuPr,convpPr,convdivPr,convdivPrLoc,convmaxdivPr;

int iters = 3;
for(int i=0; i<iters; ++i) {
    Mesh Kh(nx, ny, 0., 0., d_x, d_y+globalVariable::Epsilon);

    const R h = 1./(nx-1);
    const R invh = 1./h;

    Space V0h(Kh, DataFE<Mesh>::RT0); Space Q0h(Kh, DataFE<Mesh>::P0); // for area computation (always lowest order)
    
    // Space V2h(Kh, DataFE<Mesh>::RT2); Space Q2h(Kh, DataFE<Mesh>::P2); // for the RHS
    Lagrange2 FEvelocity2(4); Space V2h(Kh, FEvelocity2); Space Q2h(Kh, DataFE<Mesh>::P2); // for the RHS

    Space Vh(Kh, DataFE<Mesh>::RT1);
    Space Qh(Kh, DataFE<Mesh>::P1dc);

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

    MacroElement<Mesh> macro(Kh_i, 0.25);
    

    // Create CutFEM object
    CutFEM<Mesh2> darcy(Wh); darcy.add(Ph); 

    // Fun_h fv(Wh, fun_force);
    Fun_h fq(P2h, fun_div);
    Fun_h p0(P2h, fun_exact_p);
    Fun_h u0(W2h, fun_exact_u);

    // [Params]
    FunTest p(Ph,1), q(Ph,1), u(Wh,2), v(Wh,2);

    // [Area]
    // CutFEM<Mesh2> area_comp(P0h); FunTest q0(P0h,1);
    // area_comp.addLinear(
    // +innerProduct(1, q0)
    // , Kh_i
    // ); 
    // R area = area_comp.rhs_.sum();

    // [ASSEMBLY]
    darcy.addBilinear(
    +innerProduct(u, v)
    -innerProduct(p, div(v))
    +innerProduct(div(u), q)
    , Kh_i
    );
    darcy.addLinear(
    +innerProduct(fq.exprList(), q)
    , Kh_i
    );


    // [GHOST PENALTY]
    double uPenParam = 1e0; // 1e0 
    double pPenParam = 1e0; // 1e0
    darcy.addPatchStabilization( // [h^(2k) h^(2k)]
    +innerProduct(uPenParam*jump(u), jump(v)) 
    -innerProduct(pPenParam*jump(p), jump(div(v)))
    +innerProduct(pPenParam*jump(div(u)), jump(q))
    , Kh_i
    , macro
    );

    // [Boundary conditions]
    double penParam = 1e6;
    darcy.addBilinear(
    +innerProduct(p, v*n) 
    +innerProduct(u*n, penParam * v*n)
    , interface
    );
    darcy.addLinear(fun_exact_u,
    +innerProduct(1, penParam * v*n)
    , interface
    );
    // darcy.addLinear(
    // +innerProduct(u0*n, penParam * v*n)
    // , interface
    // );  

    darcy.addBilinear(
    +innerProduct(p, v*n) 
    +innerProduct(u*n, penParam * v*n)
    , Kh_i, INTEGRAL_BOUNDARY
    );
    darcy.addLinear(fun_exact_u,
    +innerProduct(1, penParam * v*n)
    , Kh_i, INTEGRAL_BOUNDARY
    ); 
    // darcy.addLinear(
    // +innerProduct(u0*n, penParam * v*n)
    // , Kh_i, INTEGRAL_BOUNDARY
    // ); 

    // ActiveMesh<Mesh> Kh_itf(Kh);
    // darcy.addBilinearIntersection(
    // +innerProduct(p, v*n) 
    // +innerProduct(u*n, penParam * v*n)
    // , Kh_itf, Kh_i, INTEGRAL_BOUNDARY
    // );
    // darcy.addLinearIntersection(
    // +innerProduct(u0*n, penParam * v*n)
    // , Kh_itf, Kh_i, INTEGRAL_BOUNDARY
    // );
    // Fun_h u00(Wh, fun_exact_u);
    // darcy.setDirichletHdiv(u00, Kh_i); 

    // [LAGRANGE MULT]
    R meanP = integral(Kh_i,p0.expr(),0);
    darcy.addLagrangeMultiplier(
    +innerProduct(1, p), meanP
    , Kh_i
    );

    // [Add to last diagonal entry of LHS]
    // int nb_dof = Wh.get_nb_dof()+Ph.get_nb_dof();
    // darcy.mat_[make_pair(nb_dof,nb_dof)] = area;

    matlab::Export(darcy.mat_[0], "mat"+std::to_string(i)+"Cut.dat"); 
    // nx = 2*nx-1;
    // ny = 2*ny-1;
    // continue;
    darcy.solve("umfpack");

    std::cout << "Lagrange multiplier value: " << std::endl;
    std::cout << darcy.rhs_(Wh.get_nb_dof()+Ph.get_nb_dof())<< std::endl;

    // EXTRACT SOLUTION
    int nbdof_vel = Wh.get_nb_dof();
    int nbdof_pres = Ph.get_nb_dof();
    Rn_ data_uh = darcy.rhs_(SubArray(nbdof_vel,0));
    Rn_ data_ph = darcy.rhs_(SubArray(nbdof_pres,nbdof_vel));

    Fun_h uh(Wh, data_uh);
    Fun_h ph(Ph, data_ph);

    auto uh_0dx = dx(uh.expr(0));
    auto uh_1dy = dy(uh.expr(1));        

    // [Post process pressure]
    // R meanP = integral(Kh_i,p0.expr(),0);
    // R meanPfem = integral(Kh_i,ph.expr(),0);
    // // std::cout << meanP << std::endl;
    // ph.v -= meanPfem/area;
    // ph.v += meanP/area;

    // L2 norm vel
    R errU      = L2normCut(uh,fun_exact_u,0,2);
    R errP      = L2normCut(ph,fun_exact_p,0,1);
    R errDiv    = L2normCut (uh_0dx+uh_1dy,fun_div,Kh_i);
    R maxErrDiv = maxNormCut(uh_0dx+uh_1dy,fun_div,Kh_i);

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

    if(i==0) {convpPr.push_back(0);convuPr.push_back(0);convdivPr.push_back(0);convdivPrLoc.push_back(0);convmaxdivPr.push_back(0);}
    else {
    convpPr.push_back( log(pPrint[i]/pPrint[i-1])/log(hPr[i]/hPr[i-1]));
    convuPr.push_back( log(uPrint[i]/uPrint[i-1])/log(hPr[i]/hPr[i-1]));
    convdivPr.push_back( log(divPrint[i]/divPrint[i-1])/log(hPr[i]/hPr[i-1]));
    convmaxdivPr.push_back( log(maxDivPrint[i]/maxDivPrint[i-1])/log(hPr[i]/hPr[i-1]));
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
    << std::endl;
}
}
