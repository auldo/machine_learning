#include "machine_learning.h"
#include <iostream>

int main() {
    std::cout << "linear regression" << std::endl;
    std::string path{"../datasets/diabetes.tab.txt"};
    tensor data{read_tab_separated(path)};
    std::cout << data[{3, 2}] << std::endl;
}