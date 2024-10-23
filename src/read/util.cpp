#include "read/util.h"

std::string file_to_string(const std::string& filename) {
    std::ifstream t(filename);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

std::vector<std::string> split_string(const std::string& str, const char& delim) {
    std::stringstream stream(str);
    std::vector<std::string> result{};
    std::string to;
    while(std::getline(stream,to,delim))
        result.push_back(to);
    return result;
}