#include "functions.hpp"

double Sigmoid::forward(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::backward(double x) const {
    double s = this->forward(x); // シグモイド関数の値
    return s * (1.0 - s);  // 導関数の計算
}

double Relu::forward(double x) const {
    return std::max(0.0, x);
}

double Relu::backward(double x) const {
    return (x > 0.0) ? 1.0 : 0.0;
}