#ifndef READ_UTIL_H
#define READ_UTIL_H

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

std::string file_to_string(const std::string& filename);
std::vector<std::string> split_string(const std::string& str, const char& delim);

#endif