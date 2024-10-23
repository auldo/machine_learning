#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

template <typename T>
concept arithmetic = std::integral<T> or std::floating_point<T>;

template <typename tensor_type> requires arithmetic<tensor_type>
class array {
    using array_iterator = tensor_type*;
    using const_array_iterator = const tensor_type*;
    size_t _size;
    std::unique_ptr<tensor_type[]> _data{std::make_unique<tensor_type[]>(0)};
public:
    array() : _size(0) {}
    array(std::initializer_list<tensor_type> init) : _size(init.size()), _data(std::make_unique<tensor_type[]>(init.size())) {
        for(auto i{0}; i < init.size(); ++i)
            _data[i] = *(init.begin() + i);
    }
    explicit array(const size_t size) : _size(size), _data(std::make_unique<tensor_type[]>(size)) {}

    array(const array &other) = delete;
    array &operator=(array &&other)  noexcept {
        this->reset_size(other._size);
        this->_data = std::move(other._data);
        other.reset_size(0);
        return *this;
    }

    array(array &&other) = delete;
    array &operator=(const array &other) = delete;

    array_iterator begin() { return _data.get(); }
    array_iterator end() { return _data.get() + _size; }

    [[nodiscard]] const_array_iterator begin() const { return _data.get(); }
    [[nodiscard]] const_array_iterator end() const { return _data.get() + _size; }

    void reset_size(size_t size) {
        this->_size = size;
        this->_data = std::make_unique<tensor_type[]>(size);
    }

    tensor_type& operator[](size_t idx) {
        if(idx >= this->_size)
            throw std::out_of_range("index out of range");
        return this->_data[idx];
    }

    [[nodiscard]] tensor_type multipliedSum() const {
        tensor_type sum{this->_size == 0 ? static_cast<tensor_type>(0) : static_cast<tensor_type>(1)};
        for(auto& elem : *this)
            sum *= elem;
        return sum;
    }

    [[nodiscard]] size_t size() const { return _size; }
};

template <typename tensor_type> requires arithmetic<tensor_type>
class tensor {
    array<tensor_type> _data{0};
    array<size_t> _dimensions{0};
public:
    explicit tensor(array<size_t> dimensions) {
        _dimensions = std::move(dimensions);
        _data.reset_size(_dimensions.multipliedSum());
    }

    /**
    * Idx = len(Dim2) * Dim1 + Dim2
    * Idx = Dim1 * len(Dim2) * len(Dim3) + Dim2 * len(Dim3) + Dim3
    * len(Dimx) stored in _dimensions
    * Dimx stored in indices
    */
    size_t _transformIndices(array<size_t>& indices) {
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

    tensor_type& operator[](array<size_t> indices) {
        if(indices.size() != _dimensions.size())
            throw std::invalid_argument("expected "  + std::to_string(_dimensions.size()) + " indices.");
        for(auto i{0}; i < _dimensions.size(); ++i) {
            if(indices[i] >= _dimensions[i])
                throw std::out_of_range("index out of range");
        }
        auto index{_transformIndices(indices)};
        return _data[index];
    }
};

#endif //TENSOR_H