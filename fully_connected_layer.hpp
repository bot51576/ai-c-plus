// fully_connected_layer.hpp
#ifndef FULLY_CONNECTED_LAYER_H_
#define FULLY_CONNECTED_LAYER_H_

#include "lib/matrix.hpp"
#include "neuron.hpp"

class FullyConnectedLayer {
public:
    FullyConnectedLayer(size_t input_size, size_t output_size);

    // 入力行列を受け取り、出力行列を計算
    Matrix forward(const Matrix& input) const;

    // 誤差逆伝播を行い、前の層への入力誤差を計算
    Matrix backward(const Matrix& input, const Matrix& output_error) const;
    
    // 重みを更新
    void update_weights(const Matrix& delta_weights);

    // 活性化関数のセット
    void set_activation(const std::function<double(double)>& activation);

private:
    std::vector<Neuron> neurons_;
};


#endif