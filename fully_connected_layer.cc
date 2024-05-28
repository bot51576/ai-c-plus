#include "fully_connected_layer.hpp"
#include <algorithm>

FullyConnectedLayer::FullyConnectedLayer(size_t input_size, size_t output_size) 
    : neurons_(output_size, Neuron(input_size)) {}


void FullyConnectedLayer::set_activation(const std::function<double(double)>& activation) {
    for (auto& neuron : neurons_) {
        neuron.set_activation(activation);
    }
}

Matrix FullyConnectedLayer::forward(const Matrix& input) const {
    Matrix output(input.rows(), neurons_.size());
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < neurons_.size(); ++j) {
            std::vector<double> row_data(input.cols());
            for (size_t k = 0; k < input.cols(); ++k) {
                row_data[k] = input(i, k); 
            }
            output(i, j) = neurons_[j].forward(row_data); 
        }
    }
    return output;
}

Matrix FullyConnectedLayer::backward(const Matrix& input, const Matrix& output_error) const {
    // 各ニューロンの入力誤差を格納する行列
    Matrix input_error(input.rows(), input.cols()); 

    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < neurons_.size(); ++j) {
            // 各ニューロンの重みベクトルを取得
            const std::vector<double>& weights = neurons_[j].get_weights();

            // 出力誤差を重みでスケールして入力誤差に足し込む
            for (size_t k = 0; k < input.cols(); ++k) {
                input_error(i, k) += output_error(i, j) * weights[k];
            }
        }
    }
    return input_error;
}

void FullyConnectedLayer::update_weights(const Matrix& delta_weights){
    for(size_t i = 0; i < neurons_.size(); ++i){
        // delta_weights の i 列目を std::vector<double> として取り出す
        std::vector<double> delta_weights_for_neuron(delta_weights.rows());
        for (size_t j = 0; j < delta_weights.rows(); ++j) {
            delta_weights_for_neuron[j] = delta_weights(j, i); 
        }
        neurons_[i].update_weights(delta_weights_for_neuron);
    }
}

