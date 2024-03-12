## Dependencies
#find_package(MPI)
find_package(MPI REQUIRED)


set(PARALLEL_SOURCES
  cfmpi.cpp
)

## ######################
## Build rules
## ######################
#add_library(parallel SHARED ${PARALLEL_SOURCES})
set(MPI_LIBRARIES "/usr/lib/libmpi.so")  # Adjust this path if necessary
add_library(parallel SHARED ${PARALLEL_SOURCES})


target_include_directories(parallel
                           PRIVATE 
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           ${MPI_CXX_INCLUDE_PATH})

# Link MPI libraries explicitly
target_link_libraries(parallel PRIVATE ${MPI_CXX_LIBRARIES})