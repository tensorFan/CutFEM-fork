set(SOLVER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/solver.cpp
)

set(SOLVER_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}
)

set(SOLVER_LIBRARIES)
set(MUMPS_LINK_LIBRARIES)

if(USE_UMFPACK)
    find_package(UMFPACK REQUIRED)
    list(APPEND SOLVER_INCLUDE_DIRS ${UMFPACK_INCLUDES})
    list(APPEND SOLVER_LIBRARIES ${UMFPACK_LIBRARIES})
endif()

if(USE_LAPACK)
    find_library(LAPACK_LIB NAMES lapack REQUIRED)
    find_library(BLAS_LIB   NAMES blas   REQUIRED)
    list(APPEND SOLVER_LIBRARIES ${LAPACK_LIB} ${BLAS_LIB})
endif()

if(USE_MUMPS)
    enable_language(Fortran)

    if(USE_MPI)
        find_package(MPI REQUIRED COMPONENTS CXX Fortran)
    endif()

    # These names are expected by the project's FindMUMPS.cmake.
    find_library(SCOTCH_scotch_LIBRARY
        NAMES scotch
        REQUIRED
    )
    find_library(SCOTCH_scotcherr_LIBRARY
        NAMES scotcherr
        REQUIRED
    )

    find_package(MUMPS REQUIRED)

    # Dependencies missing from the MUMPS list on Arch.
    find_library(ESMUMPS_LIB  NAMES esmumps  REQUIRED)
    find_library(SCALAPACK_LIB NAMES scalapack REQUIRED)
    find_library(LAPACK_LIB   NAMES lapack   REQUIRED)
    find_library(BLAS_LIB     NAMES blas     REQUIRED)

    list(APPEND SOLVER_INCLUDE_DIRS ${MUMPS_INCLUDES})

    list(APPEND MUMPS_LINK_LIBRARIES
        ${MUMPS_LIBRARIES}
        ${ESMUMPS_LIB}
        ${SCOTCH_scotch_LIBRARY}
        ${SCOTCH_scotcherr_LIBRARY}
        ${SCALAPACK_LIB}
        ${LAPACK_LIB}
        ${BLAS_LIB}
    )

    if(USE_MPI)
        list(APPEND SOLVER_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/solverMumpsMPI.cpp
        )
        list(APPEND MUMPS_LINK_LIBRARIES
            parallel
            MPI::MPI_CXX
            MPI::MPI_Fortran
        )
    else()
        list(APPEND SOLVER_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/solverMumps.cpp
        )
    endif()
endif()

add_library(solver SHARED ${SOLVER_SOURCES})

target_include_directories(solver
    PRIVATE
        ${SOLVER_INCLUDE_DIRS}
)

target_link_libraries(solver
    PRIVATE
        ${SOLVER_LIBRARIES}
    PUBLIC
        ${MUMPS_LINK_LIBRARIES}
)
