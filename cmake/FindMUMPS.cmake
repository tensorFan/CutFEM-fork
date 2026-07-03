# Find the double-precision MUMPS installation.
#
# Optional hints:
#   MUMPS_ROOT
#   MUMPS_DIR
#   CMAKE_PREFIX_PATH
#
# Result variables:
#   MUMPS_FOUND
#   MUMPS_INCLUDE_DIR
#   MUMPS_INCLUDES
#   MUMPS_LIBRARIES

include(FindPackageHandleStandardArgs)

set(_MUMPS_HINTS
    ${MUMPS_ROOT}
    ${MUMPS_DIR}
    $ENV{MUMPS_ROOT}
    $ENV{MUMPS_DIR}
)

find_path(MUMPS_INCLUDE_DIR
    NAMES dmumps_c.h
    HINTS ${_MUMPS_HINTS}
    PATH_SUFFIXES include include/mumps MUMPS
)

find_library(MUMPS_D_LIBRARY
    NAMES dmumps
    HINTS ${_MUMPS_HINTS}
    PATH_SUFFIXES lib lib64
)

find_library(MUMPS_COMMON_LIBRARY
    NAMES mumps_common
    HINTS ${_MUMPS_HINTS}
    PATH_SUFFIXES lib lib64
)

find_library(MUMPS_PORD_LIBRARY
    NAMES pord
    HINTS ${_MUMPS_HINTS}
    PATH_SUFFIXES lib lib64
)

find_library(MUMPS_PARMETIS_LIBRARY
    NAMES parmetis
    HINTS ${_MUMPS_HINTS}
    PATH_SUFFIXES lib lib64
)

find_library(MUMPS_ESMUMPS_LIBRARY
    NAMES esmumps
    HINTS ${_MUMPS_HINTS}
    PATH_SUFFIXES lib lib64
)

find_library(MUMPS_SCOTCH_LIBRARY
    NAMES scotch scotch-7 scotch-6
    HINTS ${_MUMPS_HINTS}
    PATH_SUFFIXES lib lib64
)

find_library(MUMPS_SCOTCHERR_LIBRARY
    NAMES scotcherr scotcherr-7 scotcherr-6
    HINTS ${_MUMPS_HINTS}
    PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(MUMPS
    REQUIRED_VARS
        MUMPS_INCLUDE_DIR
        MUMPS_D_LIBRARY
        MUMPS_COMMON_LIBRARY
)

if(MUMPS_FOUND)
    set(MUMPS_INCLUDES ${MUMPS_INCLUDE_DIR})
    set(MUMPS_LIBRARIES
        ${MUMPS_D_LIBRARY}
        ${MUMPS_COMMON_LIBRARY}
    )

    foreach(_library
            MUMPS_PORD_LIBRARY
            MUMPS_PARMETIS_LIBRARY
            MUMPS_ESMUMPS_LIBRARY
            MUMPS_SCOTCH_LIBRARY
            MUMPS_SCOTCHERR_LIBRARY)
        if(${_library})
            list(APPEND MUMPS_LIBRARIES ${${_library}})
        endif()
    endforeach()
endif()

mark_as_advanced(
    MUMPS_INCLUDE_DIR
    MUMPS_D_LIBRARY
    MUMPS_COMMON_LIBRARY
    MUMPS_PORD_LIBRARY
    MUMPS_PARMETIS_LIBRARY
    MUMPS_ESMUMPS_LIBRARY
    MUMPS_SCOTCH_LIBRARY
    MUMPS_SCOTCHERR_LIBRARY
)
