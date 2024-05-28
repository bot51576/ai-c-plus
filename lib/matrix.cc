#include "matrix.hpp"

Matrix::Matrix(int rows, int cols) : rows_(rows), cols_(cols),
                                     data_(rows * cols) {}

double& Matrix::operator()(int row, int col) {
  if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
    throw std::out_of_range("Matrix index out of range");
  }
  return data_[row * cols_ + col];
}

const double& Matrix::operator()(int row, int col) const {
  if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
    throw std::out_of_range("Matrix index out of range");
  }
  return data_[row * cols_ + col];
}

int Matrix::rows() const { return rows_; }

int Matrix::cols() const { return cols_; }

Matrix Matrix::operator+(const Matrix& other) const {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument("Matrix dimensions do not match for addition");
  }
  Matrix result(rows_, cols_);
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      result(i, j) = (*this)(i, j) + other(i, j);
    }
  }
  return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument("Matrix dimensions do not match for subtraction");
  }
  Matrix result(rows_, cols_);
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      result(i, j) = (*this)(i, j) - other(i, j);
    }
  }
  return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
  if (cols_ != other.rows_) {
    throw std::invalid_argument("Matrix dimensions do not match for multiplication");
  }
  Matrix result(rows_, other.cols_);
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < other.cols_; ++j) {
      for (int k = 0; k < cols_; ++k) {
        result(i, j) += (*this)(i, k) * other(k, j);
      }
    }
  }
  return result;
}

Matrix Matrix::operator*(double scalar) const {
  Matrix result(rows_, cols_);
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      result(i, j) = (*this)(i, j) * scalar;
    }
  }
  return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for(size_t i=0; i < rows_; i++) {
        for(size_t j=0; j < cols_; j++) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

Matrix Matrix::hadamard_product(const Matrix& other) const {
    if(rows_ != other.rows_ || cols_ != other.cols_)
        throw  std::invalid_argument("Matrix dimensions must match for Hadamard product.");
    Matrix result(rows_, cols_);
    for(size_t i=0; i < rows_; i++) {
        for(size_t j=0; j < cols_; j++) {
            result(i, j) = (*this)(i, j) * other(i, j);
        }
    }
    return result;
}

void Matrix::print() const {
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      std::cout << (*this)(i, j) << " ";
    }
    std::cout << std::endl;
  }
}