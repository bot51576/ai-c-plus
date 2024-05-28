#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <functional>
#include "./lib/functions.hpp"

class Neuron {
public:
    Neuron(size_t input_size);

    // 活性化関数オブジェクトを設定
    void set_activation(const std::function<double(double)>& activation);

    // 入力ベクトルを受け取り、ニューロンの出力を計算
    double forward(const std::vector<double>& inputs) const;

    std::vector<double> backward(double output_error, const std::vector<double>& inputs) const;
    void update_weights(const std::vector<double>& delta_weights);
    const std::vector<double>& get_weights() const;

private:
    std::vector<double> weights_;   // 重みベクトル
    std::function<double(double)> activation_; // 活性化関数
};

#endif