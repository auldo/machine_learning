#ifndef VECTOR_H
#define VECTOR_H

#define USE_RETURN [[nodiscard]]

#include <memory>
#include <string>

template <typename tensor_type> requires arithmetic<tensor_type>
class vector {
    using array_iterator = tensor_type*;
    using const_array_iterator = const tensor_type*;
    size_t _size;
    std::unique_ptr<tensor_type[]> _data{std::make_unique<tensor_type[]>(0)};
public:
    vector() : _size(0) {}
    vector(std::initializer_list<tensor_type> init) : _size(init.size()), _data(std::make_unique<tensor_type[]>(init.size())) {
        for(auto i{0}; i < init.size(); ++i)
            _data[i] = *(init.begin() + i);
    }
    explicit vector(const size_t size) : _size(size), _data(std::make_unique<tensor_type[]>(size)) {}

    vector &operator=(vector &&other)  noexcept {
        this->reset_size(other._size);
        this->_data = std::move(other._data);
        other.reset_size(0);
        return *this;
    }

    /**
    * required for using vectors as return types of functions
    */
    vector(vector &&other) noexcept: _size(other._size), _data(std::move(other._data)) {}

    vector(const vector &other) = delete;
    vector &operator=(const vector &other) = delete;

    USE_RETURN array_iterator begin() { return _data.get(); }
    USE_RETURN array_iterator end() { return _data.get() + _size; }

    USE_RETURN const_array_iterator begin() const { return _data.get(); }
    USE_RETURN const_array_iterator end() const { return _data.get() + _size; }

    void reset_size(size_t size) {
        this->_size = size;
        this->_data = std::make_unique<tensor_type[]>(size);
    }

    tensor_type& operator[](size_t idx) {
        if(idx >= this->_size)
            throw std::out_of_range("index out of range");
        return this->_data[idx];
    }

    const tensor_type& operator[](size_t idx) const {
        if(idx >= this->_size)
            throw std::out_of_range("index out of range");
        return this->_data[idx];
    }

    USE_RETURN tensor_type multiplied_sum() const {
        tensor_type sum{this->_size == 0 ? static_cast<tensor_type>(0) : static_cast<tensor_type>(1)};
        for(auto& elem : *this)
            sum *= elem;
        return sum;
    }

    USE_RETURN tensor_type multiplied_sum_last_n(const size_t n) const {
        tensor_type sum{this->_size == 0 ? static_cast<tensor_type>(0) : static_cast<tensor_type>(1)};
        for(auto i{0}; i < n; ++i) {
            sum *= this->operator[](this->_size - 1 - i);
        }
        return sum;
    }

    USE_RETURN size_t size() const { return _size; }
    USE_RETURN std::string to_string() const {
        std::string result;
        for(const auto& elem : *this) {
            result += std::to_string(elem);
        }
        return result;
    }
};

#endif