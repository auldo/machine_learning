#ifndef ARITHMETIC_H
#define ARITHMETIC_H

#include <utility>

template <typename T>
concept arithmetic = std::integral<T> or std::floating_point<T>;

#endif