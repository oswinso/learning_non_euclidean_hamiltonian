find_package(fmt REQUIRED)

add_executable(simulate
  body.cpp
  customize.cpp
  math_func.cpp
  simulate.cpp
  system.cpp
  )
target_include_directories(simulate PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(simulate PRIVATE OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS} fmt::fmt)

add_executable(check_matmul
  check_matmul.cpp
  math_func.cpp
  )
target_include_directories(check_matmul PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(check_matmul PRIVATE OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
