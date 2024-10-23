#ifndef INCLUDE_READ_TAB_SEPARATED_H
#define INCLUDE_READ_TAB_SEPARATED_H

#include "tensor.h"
#include "read/util.h"

tensor<float> read_tab_separated(const std::string& path);

#endif