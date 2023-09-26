CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)
if (COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif (COMMAND cmake_policy)

project(cppnpy)

set(LIBRARY_NAME ${PROJECT_NAME})
option(BUILD_EXAMPLES "Build example for cppnpy" OFF)

find_package(ZLIB REQUIRED)

add_library(${LIBRARY_NAME} cppnpy.cpp)
target_include_directories(${LIBRARY_NAME} PUBLIC include ${ZLIB_INCLUDE_DIRS})
target_link_libraries(cppnpy PUBLIC ${ZLIB_LIBRARIES} spdlog::spdlog)
set_property(TARGET cppnpy PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(cppnpy PRIVATE -O2)

#if (WIN32)
#    install(TARGETS ${LIBRARY_NAME}
#            EXPORT ${PROJECT_NAME}-targets
#            RUNTIME DESTINATION lib
#            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
#else ()
#    install(TARGETS ${LIBRARY_NAME}
#            EXPORT ${PROJECT_NAME}-targets
#            LIBRARY DESTINATION lib
#            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
#endif ()

#install(FILES "include/cppnpy/cppnpy.h"
#		DESTINATION include)
#install(FILES "mat2npz" "npy2mat" "npz2mat"
#		DESTINATION bin
#		PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

if (BUILD_EXAMPLES)
    add_executable(example1 example1.cpp)
    target_link_libraries(example1 cppnpy)
endif ()

## Export project export set to the build tree
#export(
#        EXPORT ${PROJECT_NAME}-targets
#        FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-targets.cmake"
#)
#
## Export the package for use from the build-tree
## (this registers the build-tree with a global CMake-registry ~/.cmake)
#export(PACKAGE ${PROJECT_NAME})
#
#include(CMakePackageConfigHelpers)
#
#set(INCLUDE_INSTALL_DIR include)
#configure_package_config_file(
#        ${PROJECT_SOURCE_DIR}/cmake/Modules/project-config.cmake.in
#        ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
#        INSTALL_DESTINATION ${PROJECT_BINARY_DIR}
#        PATH_VARS INCLUDE_INSTALL_DIR
#)
