#ifndef MATRIX_H_
#define MATRIX_H_

#include <valarray>
#include <stdexcept>
#include <iostream>

class Matrix {
    public:
    // コンストラクタ: 行数、列数を受け取り、要素を0で初期化
    Matrix(int rows, int cols);

    // 要素アクセス演算子: 行番号と列番号を指定して要素にアクセス
    double& operator()(int row, int col);

    // const版の要素アクセス演算子
    const double& operator()(int row, int col) const;

    // 行数を返す
    int rows() const;

    // 列数を返す
    int cols() const;

    // 行列の加算
    Matrix operator+(const Matrix& other) const;

    // 行列の減算
    Matrix operator-(const Matrix& other) const;

    // 行列の乗算
    Matrix operator*(const Matrix& other) const;

    // スカラー倍
    Matrix operator*(double scalar) const;

    //行列の転置
    Matrix transpose() const;

    //要素ごとの積(アダマール積)
    Matrix hadamard_product(const Matrix& other) const;

    // 行列の表示
    void print() const;

    private:
        int rows_;
        int cols_;
        std::valarray<double> data_;
};

using Vector = Matrix;

#endif