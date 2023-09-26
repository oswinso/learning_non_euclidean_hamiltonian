#include <charconv>
#include <string_view>
#include <vector>

#include "spdlog/spdlog.h"

#include "parse.h"

int main(int argc, char **argv) {
    const std::vector<std::string_view> args(argv + 1, argv + argc);

    if (args.size() == 0 || args.size() > 2) {
        spdlog::error("Expected 2 args. parse_grit PATH [WIDTH].");
        return 1;
    }

    size_t width = 30;
    if (args.size() == 2) {
        const auto &width_arg = args.at(1);
        auto result = std::from_chars(begin(width_arg), end(width_arg), width);

        if (result.ec == std::errc::invalid_argument || result.ptr != end(width_arg)) {
            spdlog::error("Invalid width {}.", width_arg);
            return 1;
        }
    }

    return parse_and_save(args.at(0), width);
}
