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
    tensor<float> data({4, 3, 5, 8});

    vector<size_t> shape = {3, 2, 4, 7};
    std::cout << data._transform_indices(shape) << std::endl;

    std::cout << data._transform_index(480).to_string() << std::endl;;
}