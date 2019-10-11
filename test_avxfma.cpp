#include "gtest/gtest.h"
#include "common.h"

#include <immintrin.h>

#include <cstdint>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

const unsigned int width = 4;
const unsigned int length = 4096; /* each item is 'width' 32-bit floating-point numbers */
const unsigned int iterations = 1000000;

TEST(TFCpp, AvxFma)
{
    __m128 a[length], b[length], c[length];

    for ( unsigned int i = 0; i < length; ++i ) {
        for ( unsigned int j = 0; j < width; ++j ) {
            a[i][j] = double(rand()) / double(RAND_MAX);
            b[i][j] = double(rand()) / double(RAND_MAX);
            c[i][j] = double(rand()) / double(RAND_MAX);
        }
    }

    double result;
    {
        AutoTimer timer(&result);
        for ( unsigned int i = 0; i < iterations; ++i ) {
            for ( unsigned int j = 0; j < length; ++j ) {
                c[j] = _mm_fmadd_ps(a[j], b[j], c[j]);
            }
        }
    }

    uint64_t numFmaOps = uint64_t(length) * iterations * width;

    const double nanoseconds_per_fma_operation = result / numFmaOps * 1000 * 1000 * 1000;

    double sum = 0;
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < width; j++)
        {
            sum += c[i][j];
        }
    }

    cout << "checksum = " << sum << endl;

    cout << "time per multiply-accumulate operation using AVX: " << nanoseconds_per_fma_operation << " nanoseconds\n";

    cout << "time for " << numFmaOps << " multiply-accumulate operations using AVX: " << result << " seconds\n";
}
