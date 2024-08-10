#pragma once

#include <immintrin.h>

#define svfloat32_t __m512
#define svbool_t __mmask16

#define svdup_f32(constant) _mm512_set1_ps(constant)
#define svadd_f32(pg, op0, op1) _mm512_add_ps(op0, op1)
#define svsub_f32(pg, op0, op1) _mm512_sub_ps(op0, op1)
#define svmul_f32(pg, op0, op1) _mm512_mul_ps(op0, op1)
#define svmla_f32(pg, add, mul0, mul1) _mm512_fmadd_ps(mul0, mul1, add)

#define svwhilelt_b32(zero, limit) _mm512_cmp_epi32_mask(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8,\
																		 9, 10, 11, 12, 13, 14, 15), \
														_mm512_set1_epi32(limit),                    \
														_MM_CMPINT_LT)
#define svst1_f32(pg, addr, src) _mm512_mask_storeu_ps(addr, pg, src)
#define svld1_f32(pg, addr) _mm512_maskz_loadu_ps(pg, addr)
#define svld1 svld1_f32
#define svst1 svst1_f32
#define svadd_f32_x svadd_f32
#define svsub_f32_x svsub_f32
#define svmul_f32_x svmul_f32
#define svmla_f32_x svmla_f32