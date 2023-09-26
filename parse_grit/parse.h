#pragma once

#include <string_view>
#include <vector>

int parse_and_save(std::string_view folder, size_t column_width);

struct BodyDatas {
    BodyDatas();

    std::vector<std::vector<double>> ps;
    std::vector<std::vector<double>> qs;
    std::vector<std::vector<double>> rots;
    std::vector<std::vector<double>> Pis;
};
