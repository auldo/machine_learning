#include "machine_learning.h"
#include <iostream>

/*
We have a function of the form: y = w*x + b
- x: matrix of [n x p]
- n: data count of learning dataset
- p: number of features
- w: vector of [p x 1]: weights per feature
*/

int main() {
    std::cout << "linear regression" << std::endl;

    tensor<float> test({4, 6, 8});

    auto index{0};
    for(const auto& _elem : test) {
        std::cout << test._transform_index(index).to_string() << std::endl;
        ++index;
    }
    vector<size_t> indices{3,5,7};
    std::cout << test._transform_indices(indices) << std::endl;
}