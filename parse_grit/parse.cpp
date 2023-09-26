#include "parse.h"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "spdlog/fmt/bundled/format.h"
#include "spdlog/fmt/bundled/ostream.h"
#include "spdlog/spdlog.h"

#include "cppnpy/cppnpy.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

using Dict = std::unordered_map<std::string, std::vector<double>>;
Dict parse_mat_file(const fs::path &path, size_t col_width);
Dict merge_dicts(std::vector<Dict> &dicts);
void save_merged_dict(Dict &merged_dict,
                      const std::vector<double> &times,
                      const fs::path &save_path,
                      size_t n_bodies);

BodyDatas::BodyDatas() : ps(3), qs(3), rots(9), Pis(3) {}

int parse_and_save(std::string_view folder_sv, size_t column_width) {
    const fs::path folder{folder_sv};

    if (!fs::exists(folder)) {
        spdlog::error("Folder {} doesn't exist!", folder);
        return 1;
    }
    if (!fs::is_directory(folder)) {
        spdlog::error("Path {} is not a folder!", folder);
        return 1;
    }

    // Read current_system.json for the true mass and intertia for each planet.
    const fs::path system_json_path = folder / "current_system.json";
    if (!(fs::exists(system_json_path) && fs::is_regular_file(system_json_path))) {
        spdlog::error("Path {} is not a file!", system_json_path);
        return 1;
    }

    std::ifstream json_file{system_json_path};
    json system_json{};
    json_file >> system_json;

    std::vector<std::string> names{};
    std::vector<double> masses{};
    std::vector<double> inertias{};

    const auto bodies = system_json.at("body");
    for (const auto &body : bodies) {
        const auto name = body.at("name").get<std::string>();
        const auto mass = body.at("mass").get<double>();
        const auto inertia = body.at("inertia_tensor").get<std::vector<double>>();

        spdlog::info("{}: {}, {}", name, mass, fmt::join(inertia, ", "));

        names.emplace_back(name);
        masses.emplace_back(mass);
        inertias.insert(end(inertias), begin(inertia), end(inertia));
    }

    // Read from each of the datamats.
    std::vector<Dict> dicts{};
    for (const auto &body : names) {
        const fs::path mat_path = folder / "data_in_mat" / body;

        if (!(fs::exists(mat_path) && fs::is_regular_file(mat_path))) {
            spdlog::error("Path {} is not a file!", mat_path);
            return 1;
        }

        dicts.emplace_back(parse_mat_file(mat_path, column_width));
    }

    // Merge all bodies together.
    Dict merged_dict = merge_dicts(dicts);
    const auto times = dicts.front()["time"];

    const auto npz_path = folder / "data.npz";

    if (fs::exists(npz_path)) {
        spdlog::error("File already exists in path {}! Exiting...", npz_path);

        return 1;
    }
    save_merged_dict(merged_dict, times, npz_path, names.size());

    spdlog::info("Saved npz to {}!", npz_path);

    return 0;
}

std::vector<std::string> parse_headers(const std::string &header_line) {
    std::vector<std::string> headers{};
    std::string header{};
    std::istringstream iss{header_line};
    while (iss >> header) {
        headers.emplace_back(header);
    }

    return headers;
}

Dict parse_mat_file(const fs::path &path, size_t col_width) {
    std::ifstream mat{path};

    // 2: Read in the header.
    std::string line{};
    std::getline(mat, line);

    if (line.size() % col_width != 0) {
        throw std::runtime_error("col_width is incorrect!");
    }

    const std::vector<std::string> headers = parse_headers(line);
    const auto n_cols = headers.size();

    // 2: Read in the mat.
    Dict data;

    while (std::getline(mat, line)) {
        std::istringstream iss{line};

        for (size_t ii = 0; ii < n_cols; ii++) {
            double val;
            iss >> val;

            const auto &header = headers[ii];
            data[header].emplace_back(val);
        }
    }

    return data;
}

/**
 * @brief Dict of (n_bodies, T)
 */
void merge_field(Dict &merged_dict,
                 const std::vector<std::string> &keys,
                 std::vector<Dict> &dicts) {
    using std::begin, std::end;

    for (const auto &key : keys) {
        auto &key_vec = merged_dict[key];

        for (auto &dict : dicts) {
            key_vec.insert(begin(key_vec), begin(dict[key]), end(dict[key]));
        }
    }
}

Dict merge_dicts(std::vector<Dict> &dicts) {
    const std::vector<std::string> p_keys{"x", "y", "z"};
    const std::vector<std::string> rot_keys{
        "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"};
    const std::vector<std::string> q_keys{"p1", "p2", "p3"};
    const std::vector<std::string> Pi_keys{"pi1", "pi2", "pi3"};

    Dict merged_dict{};
    merge_field(merged_dict, q_keys, dicts);
    merge_field(merged_dict, p_keys, dicts);
    merge_field(merged_dict, rot_keys, dicts);
    merge_field(merged_dict, Pi_keys, dicts);

    return merged_dict;
}

void save_merged_dict(Dict &merged_dict,
                      const std::vector<double> &times,
                      const fs::path &save_path,
                      size_t n_bodies) {
    const std::string save_path_str = fs::absolute(save_path);

    size_t full = merged_dict["x"].size();
    if (full % n_bodies != 0) {
        throw std::runtime_error("???");
    }
    size_t T = full / n_bodies;

    std::vector<size_t> shape{n_bodies, T};

    for (const auto &[key, arr] : merged_dict) {
        cppnpy::npz_save(save_path_str, key, arr.data(), shape, "a");
    }

    // Save times.
    cppnpy::npz_save(save_path_str, "time", times, "a");
}
