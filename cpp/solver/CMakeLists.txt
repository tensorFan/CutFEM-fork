
if(${USE_UMFPACK})
find_package(UMFPACK)
if(UMFPACK_LIBRARIES)
    include_directories(${UMFPACK_INCLUDES})
    list(APPEND LINK_LIBRARY ${UMFPACK_LIBRARIES})
    # option(USE_UMFPACK "Found the UMFpack library" ON)
else()
    message("Not Found UMFPACK")
endif()
endif()
if(${USE_LAPACK})
find_package(LAPACK)
if(LAPACK_LIBRARIES)
  include_directories(${LAPACK_INCLUDES})
  list(APPEND LINK_LIBRARY ${LAPACK_LIBRARIES})
  message("Found LAPACK")
else()
      message("Not Found LAPACK")
endif()
endif()
# if(${USE_MUMPS})
# find_package(MUMPS)
# if(MUMPS_LIBRARIES AND USE_MPI)
#   include_directories(${MUMPS_INCLUDES})
#   list(APPEND LINK_LIBRARY ${MUMPS_LIBRARIES})
#   # option(USE_MUMPS "Found the mumps library" ON)
#   message("Found MUMPS")
# else()
#   if(USE_MPI)
#     message("Not Found MUMPS")
#   elseif(MUMPS_LIBRARIES)
#     include_directories(${MUMPS_INCLUDES})
#     list(APPEND LINK_LIBRARY ${MUMPS_LIBRARIES})
#     # option(USE_MUMPS "Found the mumps library" ON)
#     message("Found seq MUMPS")
#   else()
#     message("Not Found MUMPS")
#   endif()
# endif()
# endif()
message(STATUS "=== MUMPS Configuration Debug ===")
message(STATUS "USE_MUMPS: ${USE_MUMPS}")
message(STATUS "USE_MPI: ${USE_MPI}")
if(${USE_MUMPS})
    message(STATUS "Configuring MUMPS with all dependencies...")
    
    # Find MPI first (you already have this working)
    enable_language(Fortran)
    find_package(MPI REQUIRED)
    
    # MUMPS core libraries
    set(MUMPS_CORE_LIBS
        /usr/lib/libdmumps.so
        /usr/lib/libsmumps.so
        /usr/lib/libzmumps.so
        /usr/lib/libcmumps.so
        /usr/lib/libmumps_common.so
    )
    
    # Find all the dependencies MUMPS needs
    find_library(SCOTCH_LIB NAMES scotch PATHS /usr/lib)
    find_library(SCOTCHERR_LIB NAMES scotcherr PATHS /usr/lib)
    find_library(PTSCOTCH_LIB NAMES ptscotch PATHS /usr/lib)
    find_library(PTSCOTCHERR_LIB NAMES ptscotcherr PATHS /usr/lib)
    find_library(METIS_LIB NAMES metis PATHS /usr/lib)
    find_library(SCALAPACK_LIB NAMES scalapack PATHS /usr/lib)
    find_library(BLACS_LIB NAMES blacs PATHS /usr/lib)
    find_library(BLAS_LIB NAMES blas PATHS /usr/lib)
    find_library(LAPACK_LIB NAMES lapack PATHS /usr/lib)
    find_library(GFORTRAN_LIB NAMES gfortran PATHS /usr/lib)
    find_library(PORD_LIB NAMES pord PATHS /usr/lib)
    
    # ESMUMPS libraries (from SCOTCH)
    find_library(ESMUMPS_LIB NAMES esmumps PATHS /usr/lib)
    find_library(PTESMUMPS_LIB NAMES ptesmumps PATHS /usr/lib)
    
    # Build the complete library list
    set(MUMPS_LIBRARIES ${MUMPS_CORE_LIBS})
    
    # Add dependencies in the right order
    if(PTSCOTCH_LIB)
        list(APPEND MUMPS_LIBRARIES ${PTSCOTCH_LIB})
    endif()
    if(PTSCOTCHERR_LIB)
        list(APPEND MUMPS_LIBRARIES ${PTSCOTCHERR_LIB})
    endif()
    if(SCOTCH_LIB)
        list(APPEND MUMPS_LIBRARIES ${SCOTCH_LIB})
    endif()
    if(SCOTCHERR_LIB)
        list(APPEND MUMPS_LIBRARIES ${SCOTCHERR_LIB})
    endif()
    if(ESMUMPS_LIB)
        list(APPEND MUMPS_LIBRARIES ${ESMUMPS_LIB})
    endif()
    if(PTESMUMPS_LIB)
        list(APPEND MUMPS_LIBRARIES ${PTESMUMPS_LIB})
    endif()
    if(METIS_LIB)
        list(APPEND MUMPS_LIBRARIES ${METIS_LIB})
    endif()
    if(SCALAPACK_LIB)
        list(APPEND MUMPS_LIBRARIES ${SCALAPACK_LIB})
    endif()
    if(BLACS_LIB)
        list(APPEND MUMPS_LIBRARIES ${BLACS_LIB})
    endif()
    if(PORD_LIB)
        list(APPEND MUMPS_LIBRARIES ${PORD_LIB})
    endif()
    
    # Add LAPACK, BLAS, and Fortran runtime
    list(APPEND MUMPS_LIBRARIES ${LAPACK_LIB} ${BLAS_LIB})
    if(GFORTRAN_LIB)
        list(APPEND MUMPS_LIBRARIES ${GFORTRAN_LIB})
    endif()
    
    # Add MPI libraries
    list(APPEND MUMPS_LIBRARIES ${MPI_CXX_LIBRARIES})
    
    set(MUMPS_INCLUDE_DIR "/usr/include")
    set(MUMPS_FOUND TRUE)
    
    if(USE_MPI)
        #find_library(MPI_FORTRAN_LIB NAMES mpi_mpifh PATHS /usr/lib)
        #if(NOT MPI_FORTRAN_LIB)
        #    message(FATAL_ERROR "Could not find MPI Fortran library (mpi_mpifh)")
        #endif()
        #message(STATUS "Found MPI Fortran library: ${MPI_FORTRAN_LIB}")
        list(APPEND MUMPS_LIBRARIES MPI::MPI_Fortran)

        include_directories(${MUMPS_INCLUDE_DIR})
        list(APPEND LINK_LIBRARY ${MUMPS_LIBRARIES})
        message(STATUS "Found MUMPS with MPI support")
        message(STATUS "MUMPS libraries: ${MUMPS_LIBRARIES}")
    else()
        include_directories(${MUMPS_INCLUDE_DIR})
        list(APPEND LINK_LIBRARY ${MUMPS_LIBRARIES})
        message(STATUS "Found sequential MUMPS")
    endif()
endif()


# configure_file(../../cutFEMConfig.in ../../cutFEMConfig.h)

set(SOLVER_SOURCES
${CMAKE_CURRENT_SOURCE_DIR}/solver.cpp
)

if(USE_MUMPS AND USE_MPI)
list(APPEND  SOLVER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/solverMumpsMPI.cpp)
list(APPEND LINK_LIBRARY parallel)
elseif(USE_MUMPS)
list(APPEND  SOLVER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/solverMumps.cpp)
endif()

add_library(solver SHARED ${SOLVER_SOURCES})
target_include_directories(solver
PRIVATE
"${CMAKE_CURRENT_SOURCE_DIR}"
"${CMAKE_CURRENT_BINARY_DIR}"
"${PROJECT_BINARY_DIR}"
)
target_link_libraries(solver PRIVATE ${LINK_LIBRARY})
