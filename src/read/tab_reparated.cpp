#include "read/tab_separated.h"

tensor<float> read_tab_separated(const std::string& path) {
    const auto file{file_to_string(path)};
    const auto lines{split_string(file, '\n')};

    if(lines.empty())
        throw std::invalid_argument("bad file with zero lines");

    const auto column_count{split_string(lines[0], '\t').size()};

    tensor<float> result({lines.size() - 1, column_count});

    for(size_t line_index{1}; line_index < lines.size(); ++line_index) {
        const auto& line{lines[line_index]};
        const auto columns{split_string(line, '\t')};
        if(columns.size() != column_count)
            throw std::exception{};
        for(size_t column_index{0}; column_index < columns.size(); ++column_index) {
            const auto& column{columns[column_index]};
            result[{line_index - 1, column_index}] = std::stof(column);
        }
    }

    return result;
}