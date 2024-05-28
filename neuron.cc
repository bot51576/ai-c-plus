#include "neuron.hpp"
#include <random>

Neuron::Neuron(size_t input_size) : weights_(input_size + 1), // +1 for bias
                                  activation_([](double x) { return x; }) { // デフォルトは恒等関数
    // 重みとバイアスの初期化 (例: Xavierの初期値)
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<> distribution(0.0, std::sqrt(2.0 / (input_size + 1)));

    for (auto& weight : weights_) {
        weight = distribution(engine);
    }
}

void Neuron::set_activation(const std::function<double(double)>& activation) {
    activation_ = activation;
}

double Neuron::forward(const std::vector<double>& inputs) const {
    double u = weights_.back(); // バイアス項
    for (size_t i = 0; i < inputs.size(); ++i) {
        u += weights_[i] * inputs[i];
    }
    return activation_(u);
}

std::vector<double> Neuron::backward(double output_error, const std::vector<double>& inputs) const {
    // 入力値の数だけ勾配を計算
    std::vector<double> gradients(inputs.size() + 1); // +1 for bias
    
    // 各入力値に対する勾配を計算
    for (size_t i = 0; i < inputs.size(); ++i) {
        gradients[i] = output_error * inputs[i];
    }

    // バイアス項に対する勾配
    gradients.back() = output_error; 

    return gradients;
}

void Neuron::update_weights(const std::vector<double>& delta_weights) {
    // 重みとバイアスの更新
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= delta_weights[i]; // 学習率をかける処理は、上位のクラスで行う
    }
}

const std::vector<double>& Neuron::get_weights() const{
    return weights_;
}