set(PARALLEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cfmpi.cpp
)

add_library(parallel SHARED
    ${PARALLEL_SOURCES}
)

target_include_directories(parallel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(USE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)

    target_link_libraries(parallel
        PUBLIC
            MPI::MPI_CXX
    )
endif()