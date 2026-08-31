#pragma once

#include <Core/Types.h>


namespace DB::RandomSparseTransform
{

inline UInt64 splitmix64Next(UInt64 & state)
{
    state += 0x9E3779B97F4A7C15ULL;
    UInt64 z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

template <typename Compute>
void sparseScalar(Compute * data, UInt64 seed, size_t d)
{
    Compute ep = static_cast<Compute>(1 / std::sqrt(d));
    Compute k = 1 / std::sqrt(ep);
    const Compute signs[2] = {-k, k};
    UInt64 state = seed;

    UInt64 threshold = static_cast<UInt64>(18446744073709551616.0 / std::sqrt(d));

    for (size_t i = 0; i < d; ++i)
    {
        UInt64 rand = splitmix64Next(state);
        Compute val = signs[rand & 1];
        data[i] = (rand < threshold) ? val : static_cast<Compute>(0);
    }
}
}
