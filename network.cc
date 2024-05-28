#include "network.hpp"
#include <iostream>

void Network::add_layer(const FullyConnectedLayer& layer) {
    layers_.push_back(layer);
}

Matrix Network::predict(const Matrix& input) const {
    Matrix output = input;
    for (const auto& layer : layers_) {
        output = layer.forward(output);
    }
    return output;
}

    void Network::train(const Matrix& inputs, const Matrix& targets, double learning_rate, size_t epochs) {
        std::cout << "Start Train" << std::endl;
        for (size_t epoch = 0; epoch < epochs; epoch++) {
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

double Network::mse(const Matrix& outputs, const Matrix& targets) const {
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < outputs.rows(); ++i) {
        for (size_t j = 0; j < outputs.cols(); ++j) {
            sum_squared_error += std::pow(outputs(i, j) - targets(i, j), 2);
        }
    }
    return sum_squared_error / outputs.rows(); 
}

Matrix Network::mse_gradient(const Matrix& outputs, const Matrix& targets) const {
    Matrix gradient = (outputs - targets) * (2.0 / outputs.rows());
    return gradient; 
}