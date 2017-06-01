 
#pragma once

#include <cstdint>

#ifdef WITH_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif

// my own typedef just big enough to hold the 800 data points each thread will be handling
using my_size_t = int32_t;

union point_t
{
    struct
    {
        real_t z;
        real_t force;
    };
    struct
    {
        my_size_t n;
    };
};
