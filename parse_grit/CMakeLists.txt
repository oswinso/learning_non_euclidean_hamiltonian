cmake_minimum_required(VERSION 3.21)
project(parse_grit)

set(CMAKE_CXX_STANDARD 17)

find_package(nlohmann_json REQUIRED)

add_subdirectory(external/spdlog)
add_subdirectory(external/cppnpy)

add_executable(parse_grit main.cpp parse.cpp)
target_link_libraries(parse_grit PRIVATE cppnpy spdlog)
