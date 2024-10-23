#ifndef TENSOR_H
#define TENSOR_H

#define USE_RETURN [[nodiscard]]

#include <stdexcept>
#include <string>
#include "vector.h"

template <typename tensor_type> requires arithmetic<tensor_type>
class tensor {

    using tensor_iterator = tensor_type*;
    using const_tensor_iterator = const tensor_type *;

    vector<tensor_type> _data{0};
    vector<size_t> _dimensions{0};
public:
    explicit tensor(vector<size_t> dimensions) {
        _dimensions = std::move(dimensions);
        _data.reset_size(_dimensions.multiplied_sum());
    }
    explicit tensor(const tensor_type& scalar): _dimensions({1}), _data({scalar}) {}
    tensor& operator=(const tensor_type& scalar) {
        this->_dimensions = {1};
        this->_data = {scalar};
        return *this;
    }

    USE_RETURN size_t _transform_indices(vector<size_t>& indices) const {
        size_t index{0};
        for(auto i{0}; i < indices.size(); ++i) {
            auto idx{indices.size() - i - 1};
            auto sum{1};
            for(auto i2{idx+1}; i2 < indices.size(); ++i2) {
                sum *= _dimensions[i2];
            }
            index += (indices[idx] * sum);
        }
        return index;
    }

    USE_RETURN vector<size_t> _transform_index(size_t index) const {
        if(index > max_index())
            throw std::out_of_range("index out of range");
        vector<size_t> result(_dimensions.size());
        size_t last_n{_dimensions.size() - 1}; //3
        while(last_n > 0) {
            size_t prod{_dimensions.multiplied_sum_last_n(last_n)};
            size_t divisor{index / prod};
            index = index - prod * divisor;
            result[_dimensions.size() - 1 - last_n] = divisor;
            --last_n;
        }
        result[_dimensions.size() - 1] = index % _dimensions[_dimensions.size() - 1];
        return result;
    }

    tensor_type& operator[](vector<size_t> indices) {
        if(indices.size() != _dimensions.size())
            throw std::invalid_argument("expected "  + std::to_string(_dimensions.size()) + " indices.");
        for(auto i{0}; i < _dimensions.size(); ++i) {
            if(indices[i] >= _dimensions[i])
                throw std::out_of_range("index out of range");
        }
        auto index{_transform_indices(indices)};
        return _data[index];
    }

    USE_RETURN size_t rank() const {
        if(_dimensions.size() == 1 && _data.size() == 1)
            return 0;
        return _dimensions.size();
    }

    USE_RETURN tensor_type& scalar_value() {
        if(rank() != 0)
            throw std::invalid_argument("rank must be 0");
        return this->operator[]({0});
    }

    USE_RETURN size_t max_index() const {
        return _dimensions.multiplied_sum() - 1;
    }

    void resize(vector<size_t>&& dimensions) {
        if(_dimensions.multiplied_sum() != dimensions.multiplied_sum())
            throw std::invalid_argument("can't resize tensor");
        _dimensions = std::move(_dimensions);
    }

    USE_RETURN tensor_iterator begin() { return _data.begin(); }
    USE_RETURN tensor_iterator end() { return _data.end(); }

    USE_RETURN const_tensor_iterator begin() const { return _data.begin(); }
    USE_RETURN const_tensor_iterator end() const { return _data.end(); }
};

#endif //TENSOR_H