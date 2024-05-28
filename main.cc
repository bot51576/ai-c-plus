#include "./lib/matrix.hpp"

int main() {
  // 行列のインスタンス化
  Matrix mat1(2, 3);
  mat1(0, 0) = 1; mat1(0, 1) = 2; mat1(0, 2) = 3;
  mat1(1, 0) = 4; mat1(1, 1) = 5; mat1(1, 2) = 6;

  Matrix mat2(3, 2);
  mat2(0, 0) = 7; mat2(0, 1) = 8;
  mat2(1, 0) = 9; mat2(1, 1) = 10;
  mat2(2, 0) = 11; mat2(2, 1) = 12;

  // 行列の加算
  Matrix mat3 = mat1 + mat1;
  std::cout << "mat1 + mat1:" << std::endl;
  mat3.print();

  // 行列の減算
  Matrix mat4 = mat1 - mat1;
  std::cout << "mat1 - mat1:" << std::endl;
  mat4.print();

  // 行列の乗算
  Matrix mat5 = mat1 * mat2;
  std::cout << "mat1 * mat2:" << std::endl;
  mat5.print();

  // スカラー倍
  Matrix mat6 = mat1 * 2.0;
  std::cout << "mat1 * 2.0:" << std::endl;
  mat6.print();

  //要素ごとの積
  Matrix mat7 = mat1.hadamard_product(mat1);
  std::cout << "mat1 @ mat1:" << std::endl;
  mat7.print();

  return 0;
}
