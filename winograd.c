#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_sve.h>
#include "common.h"

ALWAYS_INLINE void filter_KCHW_to_KHWC(float* __restrict__ filter, int K, int C, float* __restrict__ filer_KHWC) {
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)            // 注意「伪共享」
  for(int k = 0; k < K; ++k)
    for(int h = 0; h < FLT_HW; ++h)
      for(int w = 0; w < FLT_HW; ++w)
        for(int c = 0; c < C; ++c)
          filer_KHWC[k * 3 * 3 * C + h * 3 * C + w * C + c] 
                    = filter[k * C * 3 * 3 + c * 3 * 3 + h * 3 + w];
}

ALWAYS_INLINE void u_arr_K66C_to_KC66(float* __restrict__ u_arr, int K, int C, float* __restrict__ u_arr_out) {
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for(int k = 0; k < K; ++k)
    for(int h = 0; h < TILE_IN_H; ++h)
      for(int w = 0; w < TILE_IN_W; ++w) 
        for(int c = 0; c < C; ++c)
          u_arr_out[k * C * 6 * 6 + c * 6 * 6 + h * 6 + w] 
                    = u_arr[k * 6 * 6 * C + h * 6 * C + w * C + c];
}

ALWAYS_INLINE void Image_NCHW_to_NHWC(float* __restrict__ Image, int N, int C, int H, int W, float* __restrict__ Image_NHWC) {
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for(int n = 0; n < N; ++n)
    for(int h = 0; h < H; ++h)
      for(int w = 0; w < W; ++w)
        for(int c = 0; c < C; ++c)
          Image_NHWC[n * H * W * C + h * W * C + w * C + c] 
                    = Image[n * C * H * W + c * H * W + h * W + w];
}

// trancform the (k, c) filter from filer_KHWC, store to u_arr[K][C][6][6]
ALWAYS_INLINE void filter_transform_sve(float* __restrict__ filer_KHWC,
                                    int C, float* __restrict__ u_arr, 
                                    int k, int c
                                  ) {
  float* filter_ptr = filer_KHWC + k * 3 * 3 * C;
  float* u_arr_ptr = u_arr + k * 6 * 6 * C;
  float tmp[6][3][FP32_PER_REG] ATTRIBUTE_ALIGN(128);
  DECLARE_SVE_FP32_REGS();
  z25 = svdup_f32(1.0f);
  z26 = svdup_f32( -1.0f / 6.0f  );
  z27 = svdup_f32( -1.0f / 12.0f );
  z28 = svdup_f32(  1.0f / 4.0f  );
  z29 = svdup_f32(  1.0f / 6.0f  );
  z30 = svdup_f32(  1.0f / 12.0f );
  z31 = svdup_f32(  1.0f / 24.0f  );
  svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, C-c));
  // G * filter
  for (int i = 0; i < FLT_HW; ++i) {     // 这个循环按row遍历filter（按G以及结果的column）， 按列产生结果。
    z6 = svld1(pg, filter_ptr + 0 * 3 * C + i * C + c);
    
    z0 = svmul_f32_x(pg, z28, z6);
    z1 = svmul_f32_x(pg, z26, z6);
    z2 = svmul_f32_x(pg, z26, z6);
    z3 = svmul_f32_x(pg, z31, z6);
    z4 = svmul_f32_x(pg, z31, z6);

    z6 = svld1(pg, filter_ptr + 1 * 3 * C + i * C + c);
    
    // z0 += 0;
    z1 = svmla_f32_x(pg, z1, z26, z6);
    z2 = svmla_f32_x(pg, z2, z29, z6);
    z3 = svmla_f32_x(pg, z3, z30, z6);
    z4 = svmla_f32_x(pg, z4, z27, z6);
    // z5 += 0;
    z6 = svld1(pg, filter_ptr + 2 * 3 * C + i * C + c);
    
    // z0 += 0;
    z1 = svmla_f32_x(pg, z1, z26, z6);
    z2 = svmla_f32_x(pg, z2, z26, z6);
    z3 = svmla_f32_x(pg, z3, z29, z6);
    z4 = svmla_f32_x(pg, z4, z29, z6);
    z5 = z6;

    svst1_f32(pg, tmp[0][i], z0);
    svst1_f32(pg, tmp[1][i], z1);
    svst1_f32(pg, tmp[2][i], z2);
    svst1_f32(pg, tmp[3][i], z3);
    svst1_f32(pg, tmp[4][i], z4);
    svst1_f32(pg, tmp[5][i], z5);
  }
  // (G * filter) * G_T
  for (int i = 0; i < TILE_IN_H; ++i) {    // 这个循环按row遍历(G * filter)（按G_T的column遍历），按行产生结果。
    z6 = svld1(pg, tmp[i][0]);
    
    z0 = svmul_f32_x(pg, z28, z6);
    z1 = svmul_f32_x(pg, z26, z6);
    z2 = svmul_f32_x(pg, z26, z6);
    z3 = svmul_f32_x(pg, z31, z6);
    z4 = svmul_f32_x(pg, z31, z6);

    z6 = svld1(pg, tmp[i][1]);
    
    // z0 += 0;
    z1 = svmla_f32_x(pg, z1, z26, z6);
    z2 = svmla_f32_x(pg, z2, z29, z6);
    z3 = svmla_f32_x(pg, z3, z30, z6);
    z4 = svmla_f32_x(pg, z4, z27, z6);
    // z5 += 0;
    z6 = svld1(pg, tmp[i][2]);
    
    // z0 += 0;
    z1 = svmla_f32_x(pg, z1, z26, z6);
    z2 = svmla_f32_x(pg, z2, z26, z6);
    z3 = svmla_f32_x(pg, z3, z29, z6);
    z4 = svmla_f32_x(pg, z4, z29, z6);
    z5 = z6;

    svst1_f32(pg, u_arr_ptr + i * 6 * C + 0 * C + c, z0);
    svst1_f32(pg, u_arr_ptr + i * 6 * C + 1 * C + c, z1);
    svst1_f32(pg, u_arr_ptr + i * 6 * C + 2 * C + c, z2);
    svst1_f32(pg, u_arr_ptr + i * 6 * C + 3 * C + c, z3);
    svst1_f32(pg, u_arr_ptr + i * 6 * C + 4 * C + c, z4);
    svst1_f32(pg, u_arr_ptr + i * 6 * C + 5 * C + c, z5);
  }
}

ALWAYS_INLINE void filter_transform_all(float* __restrict__ filer_KHWC, int K,  int C, float* __restrict__ u_arr) {
  PRAGMA_OMP_PARALLEL_FOR()
  for (int k = 0; k < K; ++k) {
    for(int c = 0; c < C; c += FP32_PER_REG) {
      filter_transform_sve(filer_KHWC, C, u_arr, k, c);
    }
  }
}

ALWAYS_INLINE void src_transform_sve(float* __restrict__ image_NHWC, int C, int inHeight, int inWidth, 
                                    int n, int c, int y, int x, float V[TILE_IN_H][TILE_IN_W][FP32_PER_REG]
                                    ) {
  DECLARE_SVE_FP32_REGS();
  z22 = svdup_f32( -8.0f );
  z23 = svdup_f32(  8.0f );
  z24 = svdup_f32(  1.0f );
  z25 = svdup_f32( -1.0f );
  z26 = svdup_f32(  2.0f );
  z27 = svdup_f32( -2.0f );
  z28 = svdup_f32(  4.0f );
  z29 = svdup_f32( -4.0f );
  z30 = svdup_f32(  5.0f );
  z31 = svdup_f32( -5.0f );

  svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, C-c));

  for (int xx = 0; xx < TILE_IN_W; ++xx) {   // 按列产生结果。
    z6 = svld1(pg, image_NHWC + n * inHeight * inWidth * C + (y * 4 + 0) * inWidth * C + (x * 4 + xx) * C + c);

    z0 = svmul_f32_x(pg, z28, z6);

    z6 = svld1(pg, image_NHWC + n * inHeight * inWidth * C + (y * 4 + 1) * inWidth * C + (x * 4 + xx) * C + c);

    z1 = svmul_f32_x(pg, z29, z6);
    z2 = svmul_f32_x(pg, z28, z6);
    z3 = svmul_f32_x(pg, z27, z6);
    z4 = svmul_f32_x(pg, z26, z6);
    z5 = svmul_f32_x(pg, z28, z6);

    z6 = svld1(pg, image_NHWC + n * inHeight * inWidth * C + (y * 4 + 2) * inWidth * C + (x * 4 + xx) * C + c);

    z0 = svmla_f32_x(pg, z0, z31, z6);
    z1 = svmla_f32_x(pg, z1, z29, z6);
    z2 = svmla_f32_x(pg, z2, z29, z6);
    z3 = svmla_f32_x(pg, z3, z25, z6);
    z4 = svmla_f32_x(pg, z4, z25, z6);

    z6 = svld1(pg, image_NHWC + n * inHeight * inWidth * C + (y * 4 + 3) * inWidth * C + (x * 4 + xx) * C + c);

    z1 = svmla_f32_x(pg, z1, z24, z6);
    z2 = svmla_f32_x(pg, z2, z25, z6);
    z3 = svmla_f32_x(pg, z3, z26, z6);
    z4 = svmla_f32_x(pg, z4, z27, z6);
    z5 = svmla_f32_x(pg, z5, z31, z6);

    z6 = svld1(pg, image_NHWC + n * inHeight * inWidth * C + (y * 4 + 4) * inWidth * C + (x * 4 + xx) * C + c);
    
    z0 = svmla_f32_x(pg, z0, z24, z6);
    z1 = svmla_f32_x(pg, z1, z24, z6);
    z2 = svmla_f32_x(pg, z2, z24, z6);
    z3 = svmla_f32_x(pg, z3, z24, z6);
    z4 = svmla_f32_x(pg, z4, z24, z6);

    z6 = svld1(pg, image_NHWC + n * inHeight * inWidth * C + (y * 4 + 5) * inWidth * C + (x * 4 + xx) * C + c);

    z5 = svmla_f32_x(pg, z5, z24, z6);

    svst1_f32(pg, V[0][xx], z0);
    svst1_f32(pg, V[1][xx], z1);
    svst1_f32(pg, V[2][xx], z2);
    svst1_f32(pg, V[3][xx], z3);
    svst1_f32(pg, V[4][xx], z4);
    svst1_f32(pg, V[5][xx], z5);
  }

  for (int yy = 0; yy < TILE_IN_H; ++yy) {   // 按行产生结果。
    z6 = svld1(pg, V[yy][0]);

    z0 = svmul_f32_x(pg, z28, z6);

    z6 = svld1(pg, V[yy][1]);

    z1 = svmul_f32_x(pg, z29, z6);
    z2 = svmul_f32_x(pg, z28, z6);
    z3 = svmul_f32_x(pg, z27, z6);
    z4 = svmul_f32_x(pg, z26, z6);
    z5 = svmul_f32_x(pg, z28, z6);

    z6 = svld1(pg, V[yy][2]);

    z0 = svmla_f32_x(pg, z0, z31, z6);
    z1 = svmla_f32_x(pg, z1, z29, z6);
    z2 = svmla_f32_x(pg, z2, z29, z6);
    z3 = svmla_f32_x(pg, z3, z25, z6);
    z4 = svmla_f32_x(pg, z4, z25, z6);

    z6 = svld1(pg, V[yy][3]);

    z1 = svmla_f32_x(pg, z1, z24, z6);
    z2 = svmla_f32_x(pg, z2, z25, z6);
    z3 = svmla_f32_x(pg, z3, z26, z6);
    z4 = svmla_f32_x(pg, z4, z27, z6);
    z5 = svmla_f32_x(pg, z5, z31, z6);

    z6 = svld1(pg, V[yy][4]);
    
    z0 = svmla_f32_x(pg, z0, z24, z6);
    z1 = svmla_f32_x(pg, z1, z24, z6);
    z2 = svmla_f32_x(pg, z2, z24, z6);
    z3 = svmla_f32_x(pg, z3, z24, z6);
    z4 = svmla_f32_x(pg, z4, z24, z6);

    z6 = svld1(pg, V[yy][5]);

    z5 = svmla_f32_x(pg, z5, z24, z6);

    svst1_f32(pg, V[yy][0], z0);
    svst1_f32(pg, V[yy][1], z1);
    svst1_f32(pg, V[yy][2], z2);
    svst1_f32(pg, V[yy][3], z3);
    svst1_f32(pg, V[yy][4], z4);
    svst1_f32(pg, V[yy][5], z5);
  }
}

ALWAYS_INLINE void copy_filter(float* __restrict__ u_arr, int C, int k, int c, float U[TILE_IN_H][TILE_IN_W][FP32_PER_REG]) {
  for(int yy = 0; yy < TILE_IN_H; ++yy) {
    for(int xx = 0; xx < TILE_IN_W; ++xx) {
      float* pos = u_arr + k * TILE_IN_H * TILE_IN_W * C + yy * TILE_IN_W * C + xx * C + c;
      svst1_f32(svptrue_b32(), &U[yy][xx][0], svld1(svptrue_b32(), pos));
    }
  }
}

ALWAYS_INLINE void dest_tranform_sve(float M[TILE_IN_H][TILE_IN_W][FP32_PER_REG], int K, int kk, float Y[TILE_OUT_H][TILE_IN_W][FP32_PER_REG]) {
  DECLARE_SVE_FP32_REGS()
  z22 = svdup_f32( -8.0f );
  z23 = svdup_f32(  8.0f );
  z24 = svdup_f32(  1.0f );
  z25 = svdup_f32( -1.0f );
  z26 = svdup_f32(  2.0f );
  z27 = svdup_f32( -2.0f );
  z28 = svdup_f32(  4.0f );
  z29 = svdup_f32( -4.0f );
  z30 = svdup_f32(  5.0f );
  z31 = svdup_f32( -5.0f );
  svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, K - kk));
  for (int xx = 0; xx < TILE_IN_W; ++xx) {   // 按列产生结果。
    z4 = svld1(pg, &M[0][xx][0]);
    
    z0 = z4;

    z4 = svld1(pg, &M[1][xx][0]);
    
    z0 = svadd_f32_x(pg, z0, z4);
    z1 = z4;
    z2 = z4;
    z3 = z4;
    
    z4 = svld1(pg, &M[2][xx][0]);
    
    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svsub_f32_x(pg, z1, z4);
    z2 = svadd_f32_x(pg, z2, z4);
    z3 = svsub_f32_x(pg, z3, z4);

    z4 = svld1(pg, &M[3][xx][0]);

    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svmla_f32_x(pg, z1, z26, z4);
    z2 = svmla_f32_x(pg, z2, z28, z4);
    z3 = svmla_f32_x(pg, z3, z23, z4);

    z4 = svld1(pg, &M[4][xx][0]);

    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svmla_f32_x(pg, z1, z27, z4);
    z2 = svmla_f32_x(pg, z2, z28, z4);
    z3 = svmla_f32_x(pg, z3, z22, z4);

    z4 = svld1(pg, &M[5][xx][0]);

    z3 = svadd_f32_x(pg, z3, z4);

    svst1_f32(pg, &Y[0][xx][0], z0);
    svst1_f32(pg, &Y[1][xx][0], z1);
    svst1_f32(pg, &Y[2][xx][0], z2);
    svst1_f32(pg, &Y[3][xx][0], z3);
  }

  for (int yy = 0; yy < TILE_OUT_HW; ++yy) {   // 按行产生结果。
    z4 = svld1(pg, &Y[yy][0][0]);
    
    z0 = z4;

    z4 = svld1(pg, &Y[yy][1][0]);
    
    z0 = svadd_f32_x(pg, z0, z4);
    z1 = z4;
    z2 = z4;
    z3 = z4;
    
    z4 = svld1(pg, &Y[yy][2][0]);
    
    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svsub_f32_x(pg, z1, z4);
    z2 = svadd_f32_x(pg, z2, z4);
    z3 = svsub_f32_x(pg, z3, z4);

    z4 = svld1(pg, &Y[yy][3][0]);

    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svmla_f32_x(pg, z1, z26, z4);
    z2 = svmla_f32_x(pg, z2, z28, z4);
    z3 = svmla_f32_x(pg, z3, z23, z4);

    z4 = svld1(pg, &Y[yy][4][0]);

    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svmla_f32_x(pg, z1, z27, z4);
    z2 = svmla_f32_x(pg, z2, z28, z4);
    z3 = svmla_f32_x(pg, z3, z22, z4);  // BUG Here 

    z4 = svld1(pg, &Y[yy][5][0]);

    z3 = svadd_f32_x(pg, z3, z4);

    svst1_f32(pg, &Y[yy][0][0], z0);
    svst1_f32(pg, &Y[yy][1][0], z1);
    svst1_f32(pg, &Y[yy][2][0], z2);
    svst1_f32(pg, &Y[yy][3][0], z3);
  }
}

ALWAYS_INLINE void dest_store(float Y[TILE_OUT_H][TILE_IN_W][FP32_PER_REG], float* __restrict__ out, int K, int outHeight, int outWidth, int n, int kk, int y, int x) {
  for(int k = kk; k < kk + MIN(FP32_PER_REG, K - kk); ++k)
    for(int yy = 0; yy < TILE_OUT_H; ++yy)
      for(int xx = 0; xx < TILE_OUT_W; ++xx)
        out[(long)((n * K + k) * outHeight + y * 4 + yy) * outWidth + x * 4 + xx] = Y[yy][xx][k - kk];
}

ALWAYS_INLINE void hadamard_product(float U[TILE_IN_H][TILE_IN_W][FP32_PER_REG],
                                    float V[TILE_IN_H][TILE_IN_W][FP32_PER_REG],
                                    int kk, int k,
                                    int C, int c,
                                    float M[TILE_IN_H][TILE_IN_W][FP32_PER_REG]
                                    ) {
  for(int yy = 0; yy < TILE_IN_H; ++yy) {
    for(int xx = 0; xx < TILE_IN_W; ++xx) {
      DECLARE_SVE_FP32_REGS()
      svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, C-c));
      z0 = svld1(pg, &V[yy][xx][0]);
      z1 = svld1(pg, &U[yy][xx][0]);
      z3 = svmul_f32_x(pg, z0, z1);
      M[yy][xx][k - kk] += svaddv_f32(pg, z3); //reduce
    }
  }
}

// User API for winograd F(2,3)
// image: [batch * C * inHeight * inWidth]
// filter: [K * C * 3 * 3]
// result: [batch * K * outHeight * outWidth]
void winconv_2x3(float *__restrict__ image, const int inHeight,
                 const int inWidth, const int C, float *__restrict__ filter,
                 const int K, const int N, float *__restrict__ out,
                 float *__restrict__ U, float *__restrict__ V,
                 float *__restrict__ M) {

  // m = 4; r = 3; alpha = 4
  const int outHeight = inHeight - 2;
  const int outWidth = inWidth - 2;
  const long sizeI = inHeight * inWidth;
  const int sizeF = 3 * 3;
  const int sizeO = outHeight * outWidth;
  int outgap[3] = {
      K * (inHeight - 2) * (inWidth - 2), 
      (inHeight - 2) * (inWidth - 2), 
      (inWidth - 2)
    };
  #pragma omp parallel for simd aligned(out) schedule(static)
  for(int i = 0; i < N * outgap[0]; ++i) out[i] = 0;
  // memset(out, 0, N * outgap[0] * sizeof(float));

  float* u_arr      =  (float*) aligned_alloc(ALLOC_ALIGNMENT, sizeof(float) * K * 6 * 6 * C);     
  assert(u_arr != NULL);
  float* filer_KHWC =  (float*) aligned_alloc(ALLOC_ALIGNMENT, K * C * 3 * 3 * sizeof(float));
  assert(filer_KHWC != NULL);
  float* image_NHWC =  (float*) aligned_alloc(ALLOC_ALIGNMENT, sizeof(float) * N * C * inHeight * inWidth);
  assert(image_NHWC != NULL);

  filter_KCHW_to_KHWC(filter, K, C, filer_KHWC);
  filter_transform_all(filer_KHWC, K, C, u_arr);
  Image_NCHW_to_NHWC(image, N, C, inHeight, inWidth, image_NHWC);

  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
  for (int kk = 0; kk < K; kk += FP32_PER_REG) {
    for (int n = 0; n < N; ++n) {
      for (int y = 0; y < outHeight / 4; ++y) {
        for (int x = 0; x < outWidth / 4; ++x) {
          
          float M[6][6][FP32_PER_REG] ATTRIBUTE_ALIGN(128);
          memset(M, 0, sizeof M);
          float Y[4][6][FP32_PER_REG]   ATTRIBUTE_ALIGN(128);

          for(int k = kk; k < kk + MIN(FP32_PER_REG, K - kk); ++k) {
            for (int c = 0; c < C; c += FP32_PER_REG) {
              // B_T * d
              float U[TILE_IN_H][TILE_IN_W][FP32_PER_REG] ATTRIBUTE_ALIGN(128);
              float V[TILE_IN_H][TILE_IN_W][FP32_PER_REG] ATTRIBUTE_ALIGN(128);
              copy_filter(u_arr, C, k, c, U);
              // 这里有重复运算K次
              src_transform_sve(image_NHWC, C, inHeight, inWidth, n, c, y, x, V);
              hadamard_product(U, V, kk, k, C, c, M);
            }
          }
          dest_tranform_sve(M, K, kk, Y);
          dest_store(Y, out, K, outHeight, outWidth, n, kk, y, x);
        }
      }
    }
  }

  // 边缘处理
  // TODO

  free(u_arr);
  free(image_NHWC);
  free(filer_KHWC);
}