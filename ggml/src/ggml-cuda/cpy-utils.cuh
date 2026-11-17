#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// UE4M3 encoding identical to ggml_fp32_to_ue4m3 (CPU reference): round half up on the mantissa, saturate to 0x7E.
static __device__ __forceinline__ uint8_t fp32_to_ue4m3_ref(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    x = fminf(x, 448.0f);
    const uint32_t bits = __float_as_uint(x);
    const int fp32_exp  = int((bits >> 23) & 0xFF) - 127;
    const int fp32_man  = (bits >> 20) & 0x7;
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        const int man = min(int(x * 512.0f + 0.5f), 7);
        return man < 1 ? 0 : uint8_t(man);
    }
    if (ue4m3_exp >= 15) {
        return 0x7E;
    }
    int ue4m3_man = fp32_man + int((bits >> 19) & 1);
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        if (ue4m3_exp >= 15) {
            return 0x7E;
        }
    }
    return uint8_t((ue4m3_exp << 3) | ue4m3_man);
}

// E2M1 code of v for the scale d (nearest value, ties to the lower code) and its squared error, bitwise identical to
//     nvfp4_quantize_sub in ggml-quants.c: the reference scans all 16 codes; for v >= 0 only the codes 0..7 can win,
//     for v < 0 code 0 and the negative codes 9..15, whose errors are exactly the errors of the magnitudes 1..7 for |v|
//     (negation is exact), in the same order. __fmul_rn/__fsub_rn: no FMA contraction, ties must match the CPU.
static __device__ __forceinline__ uint32_t quantize_f32_nvfp4_code(const float v, const float d, float & e2) {
    constexpr float kmag[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f}; // kvalues_mxfp4[0..7]
    const float av = fabsf(v);
    int   best_m   = 0;
    float best_err = av; // fabsf(0*d - |v|)
#pragma unroll
    for (int m = 1; m < 8; ++m) {
        const float err = fabsf(__fmul_rn(kmag[m], d) - av);
        if (err < best_err) {
            best_m   = m;
            best_err = err;
        }
    }
    const float qv = v < 0.0f ? -kmag[best_m] : kmag[best_m];
    const float e  = __fsub_rn(__fmul_rn(qv, d), v);
    e2 = __fmul_rn(e, e);
    return v < 0.0f && best_m > 0 ? 8 + best_m : best_m;
}

// One 16 value sub-block of quantize_row_nvfp4_ref (UE4M3 scale amax/6, nearest E2M1 value, ties to the lower code),
//     computed by a group of 4 consecutive lanes (lane t: values 4t..4t+3). All 4 lanes must call it (valid == false:
//     no memory accesses).
// four_over_six (opt-in, GGML_CUDA_NVFP4_FOUR_OVER_SIX=1, not the CPU reference): also try the absmax scaled to 4 and keep
//     the lower squared error (summed in j order by lane 0). Lower MSE, but worse for Gemma 4 26B-A4B K/V in practice
//     (first-token probabilities and AA-LCR thinking starts drift from f16 more than with amax/6).
template <bool four_over_six>
static __device__ __forceinline__ void quantize_f32_nvfp4_sub_x4(
        const float * __restrict__ xb, block_nvfp4 * __restrict__ y, const int s, const bool valid) {
    const int t = threadIdx.x % 4;
    float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (valid) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            x[k] = xb[4*t + k];
        }
    }
    float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 2, 4));

    const uint8_t ue6 = fp32_to_ue4m3_ref(amax / 6.0f);
    const uint8_t ue4 = fp32_to_ue4m3_ref(amax / 4.0f);
    const float   d6  = ggml_cuda_ue4m3_to_fp32(ue6);
    const float   d4  = ggml_cuda_ue4m3_to_fp32(ue4);

    uint32_t c6 = 0; // code of value 4t+k in bits 8k..8k+3
    uint32_t c4 = 0;
    float e6[4];
    float e4[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        c6 |= quantize_f32_nvfp4_code(x[k], d6, e6[k]) << (8*k);
        if constexpr (four_over_six) {
            c4 |= quantize_f32_nvfp4_code(x[k], d4, e4[k]) << (8*k);
        }
    }

    bool use4 = false;
    if constexpr (four_over_six) {
        // sequential sums over j = 0..15 in lane 0 of the group
        float sse6 = 0.0f;
        float sse4 = 0.0f;
#pragma unroll
        for (int src = 0; src < 4; ++src) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float a6 = __shfl_sync(0xFFFFFFFF, e6[k], src, 4);
                const float a4 = __shfl_sync(0xFFFFFFFF, e4[k], src, 4);
                sse6 = __fadd_rn(sse6, a6);
                sse4 = __fadd_rn(sse4, a4);
            }
        }
        use4 = __shfl_sync(0xFFFFFFFF, sse4 < sse6, 0, 4);
    }
    const uint32_t c  = use4 ? c4 : c6;
    // byte j (j < 8) of qs = code j | code (j + 8) << 4: lanes 0/1 hold codes 0..7, lanes 2/3 codes 8..15
    const uint32_t hi = __shfl_down_sync(0xFFFFFFFF, c, 2, 4);
    if (valid && t < 2) {
        const uint32_t w = c | (hi << 4);
        ggml_cuda_memcpy_1<4>(y->qs + s*(QK_NVFP4_SUB/2) + 4*t, &w);
        if (t == 0) {
            y->d[s] = use4 ? ue4 : ue6;
        }
    }
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
