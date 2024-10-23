#include "machine_learning.h"

/*
We have a function of the form: y = w*x + b
- x: matrix of [n x p]
- n: data count of learning dataset
- p: number of features
- w: vector of [p x 1]: weights per feature
*/

int main() {
    std::cout << "linear regression" << std::endl;
    tensor<float> data({4, 2, 8, 90});
    std::cout << data[{3, 1, 7, 89}] << std::endl;
}