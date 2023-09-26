#include "math_func.hpp"
#include <iostream>

int main(int argc, char **argv) {
  using namespace rb_sim;

  const Matrix<ld> A{std::vector<ld>{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3};
  const Matrix<ld> B{std::vector<ld>{1, 0, 0, 0, 2, 0, 0, 0, 3}, 3, 3};

  const Matrix<ld> C = A * B;

  std::cout << A << "\n";
  std::cout << "\n";
  std::cout << B << "\n";
  std::cout << "\n";
  std::cout << C << "\n";

  return 0;
}
