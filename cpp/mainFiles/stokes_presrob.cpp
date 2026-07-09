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


// #define PROBLEM_2026_KIKUCHI_3D_PRESROB_FITTED // Section 5.2 pressure-robustness test

// #define PROBLEM_2026_KIKUCHI_3D_PRESROB_UNFITTED // pressure-robustness test, unfitted version
// Data set for the cut-sphere Kikuchi test below.  Comment this line to recover
// the original pure-gradient test with u_exact = 0.
// #define KIKUCHI_SPHERE_USE_NONZERO_SWIRL

#define KIKUCHI_CUT_POSITION_CONDITION


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
// Purpose: pressure robustness in the genuinely cut case.  Two manufactured
// data namespaces are provided below:
//   (1) the original pure-gradient test with u_exact = 0;
//   (2) a nonzero divergence-free swirl, independent of lambda, with
//           f = curl(curl u_exact) + grad p_exact.
// In both cases
//      p_exact = lambda * (|x|^2 - 3 R^2 / 5)
// has zero mean on the exact filled ball.  Both exact velocities satisfy
//      u . n = 0,       curl(u) x n = 0
// on the sphere.  Thus velocity errors should be essentially independent of
// lambda for a pressure-robust discretisation.
//
// IMPORTANT: f is interpolated into the H(curl) velocity space before addLinear.
// This is the key pressure-robust ingredient; do not replace fh by a vector
// Lagrange/L2 interpolation when running this test.
// -----------------------------------------------------------------------------
#ifdef PROBLEM_2026_KIKUCHI_3D_PRESROB_UNFITTED

  // ---------------------------------------------------------------------------
  // Original data set: a pure pressure gradient with u_exact = 0.
  // ---------------------------------------------------------------------------
  namespace Erik_Data_KIKUCHI_3D_PRESROB_SPHERE_ZERO {

    const R radius  = 2.0/3.0;
    const R radius2 = radius*radius;
    const R eps_ls  = 1e-14;

    // Default value; override with argv[1], e.g. ./bin/stokesRT 1e7
    R pressureLambda = 1e5;

    const char* dataName = "pure-gradient data (u_exact = 0)";
    const char* dataStem = "kikuchi_sphere_zero";

    R fun_levelSet(const R3 P, const int i) {
      return P.x*P.x + P.y*P.y + P.z*P.z - radius2 + eps_ls;
    }

    R fun_exact_u(const R3 P, const int i, const int dom) {
      return 0.0;
    }

    R fun_exact_curl_u(const R3 P, const int i, const int dom) {
      return 0.0;
    }

    R fun_curlcurl_u(const R3 P, const int i, const int dom) {
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
      return fun_curlcurl_u(P, i, dom) + fun_grad_p(P, i, dom);
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

  // ---------------------------------------------------------------------------
  // Nonzero-velocity pressure-robustness data.
  //
  // Let s = R^2 - |x|^2 and choose the vector potential
  //     A = (0,0,s^3).
  // Then
  //     u = curl A = (-6 y s^2, 6 x s^2, 0)
  // is a smooth divergence-free swirl.  Moreover u = 0 and curl u = 0 on
  // |x| = R, so it satisfies the same natural sphere boundary conditions as
  // the zero-velocity test.  The pressure is kept at size lambda, while the
  // exact velocity is independent of lambda.  Thus velocity convergence curves
  // for different lambda should lie on top of each other for a robust method.
  // ---------------------------------------------------------------------------
  namespace Erik_Data_KIKUCHI_3D_PRESROB_SPHERE_SWIRL {

    const R radius  = 2.0/3.0;
    const R radius2 = radius*radius;
    const R eps_ls  = 1e-14;
    const R velocityAmplitude = 1.0;

    // Default value; override with argv[1], e.g. ./bin/stokesRT 1e7
    R pressureLambda = 1e5;

    const char* dataName = "nonzero divergence-free swirl";
    const char* dataStem = "kikuchi_sphere_swirl";

    R fun_levelSet(const R3 P, const int i) {
      return P.x*P.x + P.y*P.y + P.z*P.z - radius2 + eps_ls;
    }

    R fun_exact_u(const R3 P, const int i, const int dom) {
      const R r2 = P.x*P.x + P.y*P.y + P.z*P.z;
      const R s  = radius2 - r2;

      if (i == 0) return -6.0*velocityAmplitude*P.y*s*s;
      if (i == 1) return  6.0*velocityAmplitude*P.x*s*s;
      return 0.0;
    }

    R fun_exact_curl_u(const R3 P, const int i, const int dom) {
      const R x = P.x;
      const R y = P.y;
      const R z = P.z;
      const R s = radius2 - x*x - y*y - z*z;

      if (i == 0) return 24.0*velocityAmplitude*x*z*s;
      if (i == 1) return 24.0*velocityAmplitude*y*z*s;
      return 12.0*velocityAmplitude*s*(radius2 - 3.0*x*x - 3.0*y*y - z*z);
    }

    // Since div u = 0, curl(curl u) = -Delta u.  For the swirl above,
    // curl(curl u) is the following cubic polynomial.
    R fun_curlcurl_u(const R3 P, const int i, const int dom) {
      const R r2 = P.x*P.x + P.y*P.y + P.z*P.z;
      const R radialFactor = 7.0*r2 - 5.0*radius2;

      if (i == 0) return  24.0*velocityAmplitude*P.y*radialFactor;
      if (i == 1) return -24.0*velocityAmplitude*P.x*radialFactor;
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
      return fun_curlcurl_u(P, i, dom) + fun_grad_p(P, i, dom);
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

#ifdef KIKUCHI_SPHERE_USE_NONZERO_SWIRL
  using namespace Erik_Data_KIKUCHI_3D_PRESROB_SPHERE_SWIRL;
#else
  using namespace Erik_Data_KIKUCHI_3D_PRESROB_SPHERE_ZERO;
#endif

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
      std::cout << " data   = " << dataName << std::endl;
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
      Fun_h gradph(Uh, fun_grad_p);   // grad(p) RHS only

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

      // RHS ghost product contribution: s_h(f_h, v_h)
      stokes.addFaceStabilizationRHS(
        + innerProduct(jump(fh.exprList()), etaGhost * pow(hi,1) *jump(v)) // + innerProduct(jump(fh, 1., -1.), etaGhost * pow(hi,1) *jump(v))
        // + innerProduct(jump(gradph.exprList()), etaGhost * pow(hi,1) * jump(v))
        , Khi
        , macro
      );
      
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
        Fun_h solu(Uh, fun_exact_u);
        Fun_h solp(Ph, fun_exact_p);
        Fun_h soluErr(Uh, fun_exact_u);
        Fun_h solpErr(Ph, fun_exact_p);
        soluErr.v -= uh.v;
        solpErr.v -= ph.v;
        soluErr.v.map(fabs);
        solpErr.v.map(fabs);

        Paraview<Mesh> writer(
          Khi,
          std::string("stokes3D_") + dataStem + "_" + std::to_string(i) + ".vtk"
        );
        writer.add(uh, "velocity", 0, 3);
        writer.add(ph, "pressure", 0, 1);
        writer.add(fh, "rhs_Hcurl", 0, 3);
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

    const std::string csvName = std::string(dataStem) + "_lambda_"
      + std::to_string(pressureLambda) + "_convergence.csv";
    std::ofstream csv(csvName);
    csv << "h,err_u,rate_u,err_curl_u,rate_curl_u,err_p,rate_p,err_grad_p,rate_grad_p,err_div_u,err_max_div_u\n";
    csv << std::setprecision(16);
    for (int i = 0; i < h.size(); ++i) {
      csv << h[i] << ","
          << ul2[i] << "," << convu[i] << ","
          << curlul2[i] << "," << convcurlu[i] << ","
          << pl2[i] << "," << convp[i] << ","
          << gradpl2[i] << "," << convgradp[i] << ","
          << divl2[i] << "," << divmax[i] << "\n";
    }
    std::cout << "Wrote convergence data to " << csvName << std::endl;

    std::cout << "CPU time = " << CPUtime() - cpubegin << std::endl;
  }
#endif


#ifdef KIKUCHI_CUT_POSITION_CONDITION

namespace KikuchiCutPositionData {

const R radius  = 2.0 / 3.0;
const R radius2 = radius * radius;
const R eps_ls  = 1e-14;

// The sphere is translated only in the x direction.
R sphereShiftX = 0.0;

R fun_levelSet(const R3 P, const int i) {
  const R x = P.x - sphereShiftX;
  return x * x + P.y * P.y + P.z * P.z - radius2 + eps_ls;
}

} // namespace KikuchiCutPositionData

using namespace KikuchiCutPositionData;

struct CutQuality {
  R minCutVolume   = std::numeric_limits<R>::infinity();
  R minCutFraction = std::numeric_limits<R>::infinity();
  int activeElementIndex     = -1;
  int backgroundElementIndex = -1;
  int numberOfCutElements    = 0;
};

template <typename CutMesh>
CutQuality computeSmallestActiveCut(const CutMesh &Khi) {
  CutQuality quality;

  for (int k = 0; k < Khi.get_nb_element(); ++k) {
    if (!Khi.isCut(k, 0)) {
      continue;
    }

    ++quality.numberOfCutElements;

    // get_cut_part(k,0) is the physical part retained in the truncated active
    // mesh. Its measure is a volume in this 3D test.
    const R cutVolume     = Khi.get_cut_part(k, 0).measure();
    const R elementVolume = Khi[k].measure();
    const R cutFraction   = cutVolume / elementVolume;

    if (cutFraction < quality.minCutFraction) {
      quality.minCutVolume          = cutVolume;
      quality.minCutFraction        = cutFraction;
      quality.activeElementIndex    = k;
      quality.backgroundElementIndex = Khi.idxElementInBackMesh(k);
    }
  }

  assert(quality.numberOfCutElements > 0);
  assert(std::isfinite(quality.minCutVolume));
  assert(std::isfinite(quality.minCutFraction));
  return quality;
}

template <typename Mesh, typename CutMesh>
int assembleAndExportMatrix(
    const Mesh &Kh,
    CutMesh &Khi,
    const R h,
    const bool useMacroStabilization,
    const std::string &matrixFile) {

  typedef TestFunction<Mesh> FunTest;
  typedef FESpace3 Space;
  typedef CutFESpaceT3 CutSpace;

  Space Uh_(Kh, DataFE<Mesh>::Ned0);
  Space Ph_(Kh, DataFE<Mesh>::P1);

  CutSpace Uh(Khi, Uh_);
  CutSpace Ph(Khi, Ph_);

  CutFEM<Mesh> stokes(Uh);
  stokes.add(Ph);

  FunTest u(Uh, 3, 0), v(Uh, 3, 0);
  FunTest p(Ph, 1, 0), q(Ph, 1, 0);

  stokes.addBilinear(
      +innerProduct(curl(u), curl(v))
      +innerProduct(grad(p), v),
      Khi);

  stokes.addBilinear(
      +innerProduct(u, grad(q)),
      Khi);

  if (useMacroStabilization) {
    const R etaGhost = 1.0;
    MacroElement<Mesh> macro(Khi, 0.25);

    // Same macro ghost-product stabilization as in the uploaded pressure-
    // robustness test. No stabilized RHS is needed because only the matrix is
    // being studied here.
    stokes.addFaceStabilization(
        +innerProduct(etaGhost * pow(h, 1) * jump(curl(u)), jump(curl(v)))
        +innerProduct(etaGhost * pow(h, 1) * jump(grad(p)), jump(v))
        +innerProduct(etaGhost * pow(h, 1) * jump(u), jump(grad(q))),
        Khi,
        macro);
  }

  // Remove the constant-pressure nullspace in exactly the same way for both
  // methods.
  stokes.addLagrangeMultiplier(
      innerProduct(1, p),
      0.0,
      Khi);

  matlab::Export(stokes.mat_[0], matrixFile);

  return Uh.get_nb_dof() + Ph.get_nb_dof() + 1;
}

std::string caseTag(const int caseIndex) {
  std::ostringstream out;
  out << "case_" << std::setw(2) << std::setfill('0') << caseIndex;
  return out.str();
}

int main(int argc, char **argv) {
  typedef FunFEM<Mesh3> Fun_h;
  typedef Mesh3 Mesh;
  typedef ActiveMeshT3 CutMesh;
  typedef FESpace3 Space;

  MPIcf cfMPI(argc, argv);

  // This is the mesh from the second refinement iteration of the uploaded test:
  // nx = ny = nz = 17 and h = 1/8.
  const int nx = 17;
  const int ny = 17;
  const int nz = 17;
  const R h = 2.0 / (nx - 1);

  Mesh Kh(nx, ny, nz, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0);

  // x = 5/8 is a background-grid vertex for h = 1/8. The base translation
  // places the rightmost point of the sphere at this vertex. A positive gap
  // alpha*h then leaves a progressively smaller active sliver in neighboring
  // tetrahedra as alpha tends to zero.
  const R targetVertexX = 5.0 / 8.0;
  const std::array<R, 5> gapOverH = {
      0.006,// 0.01,// 2.0e-1,
      0.003,// 0.005,// 5.0e-2,
      0.0015,// 0.002,// 1.0e-2,
      0.00075,// 0.001,// 2.0e-3,
      0.000375};// 0.0004};// 4.0e-4};

  const std::string manifestName = "kikuchi_cut_position_manifest.csv";
  std::ofstream manifest(manifestName);
  if (!manifest) {
    std::cerr << "Could not open " << manifestName << " for writing.\n";
    return EXIT_FAILURE;
  }

  manifest
      << "case,method,nx,h,shift_x,gap_over_h,min_cut_volume,"
      << "min_cut_fraction,n_cut_elements,min_active_element,"
      << "min_background_element,matrix_size,matrix_file\n";
  manifest << std::setprecision(17);

  for (int caseIndex = 0; caseIndex < static_cast<int>(gapOverH.size()); ++caseIndex) {
    const R alpha = gapOverH[caseIndex];
    sphereShiftX = targetVertexX - radius + alpha * h;

    Space Lh_(Kh, DataFE<Mesh>::P1);
    Fun_h levelSet(Lh_, fun_levelSet);
    InterfaceLevelSet<Mesh> interface(Kh, levelSet);

    CutMesh Khi(Kh);
    // phi > 0 is outside the sphere, so sign +1 removes the exterior.
    Khi.truncate(interface, 1);

    const CutQuality quality = computeSmallestActiveCut(Khi);
    const std::string tag = caseTag(caseIndex);

    std::cout << "\n" << tag
              << ": shift_x = " << sphereShiftX
              << ", gap/h = " << alpha
              << ", min cut volume = " << quality.minCutVolume
              << ", min cut fraction = " << quality.minCutFraction
              << std::endl;

    for (int stabilized = 0; stabilized <= 1; ++stabilized) {
      const bool useMacro = (stabilized == 1);
      const std::string method = useMacro
          ? "macro_stabilization"
          : "no_stabilization";
      const std::string matrixFile =
          "kikuchi_cut_position_" + tag + "_" + method + ".dat";

      const int matrixSize = assembleAndExportMatrix(
          Kh,
          Khi,
          h,
          useMacro,
          matrixFile);

      manifest
          << caseIndex << ","
          << method << ","
          << nx << ","
          << h << ","
          << sphereShiftX << ","
          << alpha << ","
          << quality.minCutVolume << ","
          << quality.minCutFraction << ","
          << quality.numberOfCutElements << ","
          << quality.activeElementIndex << ","
          << quality.backgroundElementIndex << ","
          << matrixSize << ","
          << matrixFile << "\n";

      std::cout << "  exported " << matrixFile
                << " (size " << matrixSize << " x " << matrixSize << ")"
                << std::endl;
    }
  }

  std::cout << "\nWrote matrix manifest to " << manifestName << std::endl;
  return EXIT_SUCCESS;
}

#endif