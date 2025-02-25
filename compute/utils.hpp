#pragma once

#include <bit>
#include <array>
#include <cstdint>

template<typename Byte, typename... Args>
consteval auto bytes_to_words(Byte byte, Args... bytes) {
    constexpr auto length = sizeof...(Args) + 1;
    static_assert(length % 4 == 0, "Byte array size must be a multiple of 4");
    const auto temp_arr = std::array<std::uint8_t, length>{std::uint8_t(byte), std::uint8_t(bytes)...};
    std::array<uint32_t, length / 4> words;
    for (size_t i = 0; i < length; i += 4) {
        // Extract 4-byte chunk as a temporary array
        const auto chunk = std::array{temp_arr[i], temp_arr[i + 1], temp_arr[i + 2], temp_arr[i + 3]};
        words[i / 4] = std::bit_cast<std::uint32_t>(chunk);
    }
    return words;
}
