#ifndef NETWORK_H_
#define NETWORK_H_

#include <vector>
#include "fully_connected_layer.hpp"

class Network {
public:
    Network() = default;

    // 層を追加
    void add_layer(const FullyConnectedLayer& layer);

    // 順伝播: 入力データから予測値を出力
    Matrix predict(const Matrix& input) const;

    // 学習: 学習データを用いてネットワークの重みを更新
    void train(const Matrix& inputs, const Matrix& targets, double learning_rate, size_t epochs);

private:
    // 誤差関数の定義（ここでは平均二乗誤差）
    double mse(const Matrix& outputs, const Matrix& targets) const;

    // 平均二乗誤差の勾配を計算
    Matrix mse_gradient(const Matrix& outputs, const Matrix& targets) const;

    std::vector<FullyConnectedLayer> layers_; // 層のリスト
};

// Network クラスのメソッド定義

inline void Network::add_layer(const FullyConnectedLayer& layer) {
    layers_.push_back(layer);
}

inline Matrix Network::predict(const Matrix& input) const {
    Matrix output = input;
    for (const auto& layer : layers_) {
        output = layer.forward(output);
    }
    return output;
}

inline void Network::train(const Matrix& inputs, const Matrix& targets, double learning_rate, size_t epochs) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        Matrix outputs = inputs;
        for (auto& layer : layers_) {
            outputs = layer.forward(outputs); 
        }

        // 誤差と勾配の計算
        double loss = mse(outputs, targets);
        Matrix output_error = mse_gradient(outputs, targets);

        // 逆伝播
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            output_error = it->backward(outputs, output_error);
        }

        // 重みの更新
        for (auto& layer : layers_) {
            layer.update_weights(output_error * learning_rate);
        }

        std::cout << "Epoch: " << epoch + 1 << "/" << epochs << " - Loss: " << loss << std::endl;
    }
}

inline double Network::mse(const Matrix& outputs, const Matrix& targets) const {
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < outputs.rows(); ++i) {
        for (size_t j = 0; j < outputs.cols(); ++j) {
            sum_squared_error += std::pow(outputs(i, j) - targets(i, j), 2);
        }
    }
    return sum_squared_error / outputs.rows();
}

inline Matrix Network::mse_gradient(const Matrix& outputs, const Matrix& targets) const {
    return (outputs - targets) * (2.0 / outputs.rows());
}



#endif