#ifndef TENSOR_H
#define TENSOR_H

#define USE_RETURN [[nodiscard]]

#include <ostream>
#include <vector>
#include <string>

template <typename T>
concept arithmetic = std::integral<T> or std::floating_point<T>;

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

template <typename tensor_type> requires arithmetic<tensor_type>
class tensor {
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

    void resize(vector<size_t> dimensions) {
        if(_dimensions.multiplied_sum() != dimensions.multiplied_sum())
            throw std::invalid_argument("can't resize tensor");
        _dimensions = std::move(_dimensions);
    }
};

#endif //TENSOR_H