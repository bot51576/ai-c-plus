#include <iostream>
#include "network.hpp"
#include "fully_connected_layer.hpp"
#include <vector>
#include <functional>
#include "lib/functions.hpp"

int main() {
    // XORの学習データ
    Matrix inputs(4, 2);
    inputs(0, 0) = 0; inputs(0, 1) = 0;
    inputs(1, 0) = 0; inputs(1, 1) = 1;
    inputs(2, 0) = 1; inputs(2, 1) = 0;
    inputs(3, 0) = 1; inputs(3, 1) = 1;

    Matrix targets(4, 1);
    targets(0, 0) = 0;
    targets(1, 0) = 1;
    targets(2, 0) = 1;
    targets(3, 0) = 0;

    // ニューラルネットワークの構築
    Network network;

    // 第1層 (隠れ層): 入力2ノード、出力2ノード
    std::cout << "Done First Layer" << std::endl;
    Sigmoid sigmoid; 
    FullyConnectedLayer layer1(2, 4);
    layer1.set_activation((std::bind(&Sigmoid::forward, &sigmoid, std::placeholders::_1))); // 活性化関数にシグモイド関数
    network.add_layer(layer1);

    // 第2層 (出力層): 入力2ノード、出力1ノード
    FullyConnectedLayer layer2(4, 1);
    layer2.set_activation((std::bind(&Sigmoid::forward, &sigmoid, std::placeholders::_1))); // 活性化関数にシグモイド関数
    network.add_layer(layer2);

    // 学習
    double learning_rate = 0.1;
    size_t epochs = 10000;
    network.train(inputs, targets, learning_rate, epochs);

    // 学習結果の確認
    std::cout << "学習後の予測結果:" << std::endl;
    Matrix predictions = network.predict(inputs);
    predictions.print();

    return 0;
}