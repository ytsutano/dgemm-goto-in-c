/*
    Please include compiler name below (you may also include any other modules
    you would like to be loaded)

COMPILER=gnu

    Please include All compiler flags and libraries as you want them run. You
    can simply copy this over from the Makefile's first few lines

CC=gcc
OPT=-O3 -march=native -ffast-math -funroll-loops -ftree-vectorize
CFLAGS=-Wall -pedantic -std=gnu99 $(OPT)
MKLROOT=/opt/intel/composer_xe_2013.1.117/mkl
LDLIBS=-lrt \
    -Wl,--start-group \
        $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
        $(MKLROOT)/lib/intel64/libmkl_sequential.a \
        $(MKLROOT)/lib/intel64/libmkl_core.a \
    -Wl,--end-group \
    -lpthread -lm
*/

/*
 * Copyright (c) 2015, 2017, Yutaka Tsutano
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 */

#include <emmintrin.h>

// Block size must be a multiple of 4.
#define BLOCK_SIZE_MC 192
#define BLOCK_SIZE_KC 256

#define USING_GCC                                                              \
    __GNUC_MINOR__ > 4 && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

static inline void gepp(const int lda, const int K, const double* A,
                        const double* B, double* restrict B_pkd,
                        double* restrict C);

static inline void gebp(const int lda, const int K, const int M,
                        const double* A, const double* restrict B_pkd,
                        double* restrict C);

static inline void mul_4x4(const int lda, const int K,
                           const double* restrict A_pkd,
                           const double* restrict B_pkd,
                           double* restrict C_aux);

void square_dgemm(int lda, double* A, double* B, double* C)
{
    const int ldap = (lda + 3) & ~0x03;

    // I'm using a VLA for B_pkd which can become too large for stack
    // allocation. I should switch to malloc when lda gets large or subdivide
    // B_pkd, but this works fine with the benchmark on Trestles. So I will just
    // leave it as is.
    double __attribute__((aligned(16))) B_pkd[2 * BLOCK_SIZE_KC * ldap];

    int k = 0;
    for (; LIKELY(k < lda - BLOCK_SIZE_KC + 1); k += BLOCK_SIZE_KC) {
        gepp(lda, BLOCK_SIZE_KC, &A[k * lda], &B[k], B_pkd, C);
    }
    if (lda - k > 0) {
        gepp(lda, lda - k, &A[k * lda], &B[k], B_pkd, C);
    }
}

static inline void gepp(const int lda, const int K, const double* A,
                        const double* B, double* restrict B_pkd,
                        double* restrict C)
{
#if USING_GCC
    B_pkd = __builtin_assume_aligned(B_pkd, 16);
#endif

    // Pack B into B_pkd. We duplicate each element so that we can use SSE2
    // instructions on it later.
    for (int n = 0; LIKELY(n < lda); n += 4) {
        for (int k = 0; LIKELY(k < K); ++k) {
            for (int nn = 0; nn < 4; ++nn) {
                const double b
                        = LIKELY((n + nn < lda)) ? B[k + lda * (n + nn)] : 0.0;
                B_pkd[0 + 2 * (nn + 4 * k + K * n)] = b;
                B_pkd[1 + 2 * (nn + 4 * k + K * n)] = b;
            }
        }
    }
    __builtin_prefetch(B_pkd, 0, 3);

    int m = 0;
    for (; LIKELY(m < lda - BLOCK_SIZE_MC + 1); m += BLOCK_SIZE_MC) {
        gebp(lda, K, BLOCK_SIZE_MC, &A[m], B_pkd, &C[m]);
    }
    if (lda - m > 0) {
        gebp(lda, K, lda - m, &A[m], B_pkd, &C[m]);
    }
}

static inline void gebp(const int lda, const int K, const int M,
                        const double* A, const double* restrict B_pkd,
                        double* restrict C)
{
#if USING_GCC
    B_pkd = __builtin_assume_aligned(B_pkd, 16);
#endif

    double __attribute__((aligned(16))) A_pkd[BLOCK_SIZE_MC * BLOCK_SIZE_KC];
    double __attribute__((aligned(16))) C_aux[BLOCK_SIZE_MC * 4];

    // Pack A into A_pkd.
    for (int m = 0; LIKELY(m < M); m += 4) {
        for (int k = 0; LIKELY(k < K); ++k) {
            for (int mm = 0; mm < 4; ++mm) {
                A_pkd[mm + 4 * k + K * m] = A[m + mm + lda * k];
            }
        }
    }

    for (int n = 0; LIKELY(n < lda); n += 4) {
        for (int m = 0; LIKELY(m < M); m += 4) {
            mul_4x4(lda, K, &A_pkd[K * m], &B_pkd[K * 2 * n], &C_aux[4 * m]);
        }

        // Unpack C_aux to C.
        for (int m = 0; LIKELY(m < M); m += 4) {
            for (int mm = 0; mm < 4; ++mm) {
                for (int nn = 0; nn < 4; ++nn) {
                    const int cm = m + mm;
                    const int cn = n + nn;
                    if (LIKELY(cm < M) && LIKELY(cn < lda)) {
                        C[cm + lda * cn] += C_aux[mm + 4 * (m + nn)];
                    }
                }
            }
        }
    }
}

#define MUL_4X4_KERNEL(k, kk)                                                  \
    const __m128d reg_a0_##kk = _mm_load_pd(A_pkd + 0 + 4 * ((k) + (kk)));     \
    const __m128d reg_a1_##kk = _mm_load_pd(A_pkd + 2 + 4 * ((k) + (kk)));     \
                                                                               \
    const __m128d reg_b0_##kk = _mm_load_pd(B_pkd + 0 + 8 * ((k) + (kk)));     \
    const __m128d reg_b1_##kk = _mm_load_pd(B_pkd + 2 + 8 * ((k) + (kk)));     \
    const __m128d reg_b2_##kk = _mm_load_pd(B_pkd + 4 + 8 * ((k) + (kk)));     \
    const __m128d reg_b3_##kk = _mm_load_pd(B_pkd + 6 + 8 * ((k) + (kk)));     \
                                                                               \
    reg_c0 = _mm_add_pd(reg_c0, _mm_mul_pd(reg_a0_##kk, reg_b0_##kk));         \
    reg_c1 = _mm_add_pd(reg_c1, _mm_mul_pd(reg_a1_##kk, reg_b0_##kk));         \
    reg_c2 = _mm_add_pd(reg_c2, _mm_mul_pd(reg_a0_##kk, reg_b1_##kk));         \
    reg_c3 = _mm_add_pd(reg_c3, _mm_mul_pd(reg_a1_##kk, reg_b1_##kk));         \
    reg_c4 = _mm_add_pd(reg_c4, _mm_mul_pd(reg_a0_##kk, reg_b2_##kk));         \
    reg_c5 = _mm_add_pd(reg_c5, _mm_mul_pd(reg_a1_##kk, reg_b2_##kk));         \
    reg_c6 = _mm_add_pd(reg_c6, _mm_mul_pd(reg_a0_##kk, reg_b3_##kk));         \
    reg_c7 = _mm_add_pd(reg_c7, _mm_mul_pd(reg_a1_##kk, reg_b3_##kk));

#pragma GCC push_options
#pragma GCC optimize("Os")
static inline void mul_4x4(const int lda, const int K,
                           const double* restrict A_pkd,
                           const double* restrict B_pkd, double* restrict C_aux)
{
#if USING_GCC
    A_pkd = __builtin_assume_aligned(A_pkd, 16);
    B_pkd = __builtin_assume_aligned(B_pkd, 16);
    C_aux = __builtin_assume_aligned(C_aux, 16);
#endif

    __builtin_prefetch(C_aux + 0, 1, 3);
    __builtin_prefetch(C_aux + 8, 1, 3);

    __m128d reg_c0 = _mm_setzero_pd();
    __m128d reg_c1 = _mm_setzero_pd();
    __m128d reg_c2 = _mm_setzero_pd();
    __m128d reg_c3 = _mm_setzero_pd();
    __m128d reg_c4 = _mm_setzero_pd();
    __m128d reg_c5 = _mm_setzero_pd();
    __m128d reg_c6 = _mm_setzero_pd();
    __m128d reg_c7 = _mm_setzero_pd();

    register int k = 0;
#define MUL_4X4_LOOP_UNROLL 32
    const int KK = K - MUL_4X4_LOOP_UNROLL + 1;
    for (; LIKELY(k < KK); k += MUL_4X4_LOOP_UNROLL) {
        MUL_4X4_KERNEL(k, 0);
        MUL_4X4_KERNEL(k, 1);
        MUL_4X4_KERNEL(k, 2);
        MUL_4X4_KERNEL(k, 3);
        MUL_4X4_KERNEL(k, 4);
        MUL_4X4_KERNEL(k, 5);
        MUL_4X4_KERNEL(k, 6);
        MUL_4X4_KERNEL(k, 7);
        MUL_4X4_KERNEL(k, 8);
        MUL_4X4_KERNEL(k, 9);
        MUL_4X4_KERNEL(k, 10);
        MUL_4X4_KERNEL(k, 11);
        MUL_4X4_KERNEL(k, 12);
        MUL_4X4_KERNEL(k, 13);
        MUL_4X4_KERNEL(k, 14);
        MUL_4X4_KERNEL(k, 15);
        MUL_4X4_KERNEL(k, 16);
        MUL_4X4_KERNEL(k, 17);
        MUL_4X4_KERNEL(k, 18);
        MUL_4X4_KERNEL(k, 19);
        MUL_4X4_KERNEL(k, 20);
        MUL_4X4_KERNEL(k, 21);
        MUL_4X4_KERNEL(k, 22);
        MUL_4X4_KERNEL(k, 23);
        MUL_4X4_KERNEL(k, 24);
        MUL_4X4_KERNEL(k, 25);
        MUL_4X4_KERNEL(k, 26);
        MUL_4X4_KERNEL(k, 27);
        MUL_4X4_KERNEL(k, 28);
        MUL_4X4_KERNEL(k, 29);
        MUL_4X4_KERNEL(k, 30);
        MUL_4X4_KERNEL(k, 31);
    }
    for (; UNLIKELY(k < K); ++k) {
        MUL_4X4_KERNEL(k, 0);
    }

    _mm_store_pd(C_aux + 0, reg_c0);
    _mm_store_pd(C_aux + 2, reg_c1);
    _mm_store_pd(C_aux + 4, reg_c2);
    _mm_store_pd(C_aux + 6, reg_c3);
    _mm_store_pd(C_aux + 8, reg_c4);
    _mm_store_pd(C_aux + 10, reg_c5);
    _mm_store_pd(C_aux + 12, reg_c6);
    _mm_store_pd(C_aux + 14, reg_c7);
}
#pragma GCC pop_options
