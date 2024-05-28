#ifndef FUNCTION_H_
#define FUNCTION_H_

#include <cmath>

class Sigmoid {
    public:
        double forward(double x) const;
        double backward(double x) const;

};

class Relu {
    public:
        double forward(double x) const;
        double backward(double x) const;
};

#endif