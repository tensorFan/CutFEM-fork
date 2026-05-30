Refactored integrationFunFEM.hpp
================================

This package contains a cleaned replacement for cpp/FESpace/integrationFunFEM.hpp.

Main changes:
- Removed large blocks of dead/commented legacy code.
- Preserved the public integral(...) overloads used by the existing library.
- Centralised MPI reduction in IntegrationFunFEMDetail::mpi_sum.
- Centralised active cut-cell quadrature in for_each_active_cut_quadrature.
- Changed the 3-argument ActiveMesh expression integral to integrate over the actual active elements/domains instead of assuming domains are 0,...,get_nb_domain()-1.
- Added integralFunction(ActiveMesh, f, component, itq) for direct integration of analytic functions f(point, component, domain), avoiding an unnecessary FunFEM interpolation.

Suggested install:
1. Back up your old file:
   cp cpp/FESpace/integrationFunFEM.hpp cpp/FESpace/integrationFunFEM.hpp.bak
2. Replace it:
   cp integrationFunFEM_refactored.hpp cpp/FESpace/integrationFunFEM.hpp
3. Rebuild from a clean build directory.

Note:
I could not fully compile the project in this sandbox because MPI was unavailable during CMake configuration. Treat this as a careful drop-in refactor, but test it in your local build.
