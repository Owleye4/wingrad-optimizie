#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_sve.h>
#include "common.h"
ALWAYS_INLINE void boundaryTreatment(float *__restrict__ packedImage, ImgShape is, float *__restrict__ packedFilter, FltShape fs, float *__restrict__ out) {
  int N = is.numImg, C = is.ic, H = is.h, W = is.w, K = fs.oc;
  long inpos, knpos, outpos;
  int dimIn[4] = {N, H, W, C};
  int dimKn[4] = {K, 3, 3, C};
  int dimOut[4] = {N, K, H - 2, W - 2};

  int ingap[3] = {dimIn[1] * dimIn[2] * dimIn[3], dimIn[2] * dimIn[3],
                  dimIn[3]};
  int kngap[3] = {dimKn[1] * dimKn[2] * dimKn[3], dimKn[2] * dimKn[3],
                  dimKn[3]};
  int outgap[3] = {dimOut[1] * dimOut[2] * dimOut[3], dimOut[2] * dimOut[3],
                   dimOut[3]};

  long outHeight = H - 2;
  long outWidth = W - 2;


  if(outHeight % TILE_OUT_HW > 0){
  #pragma omp parallel for private(inpos, knpos, outpos) collapse(2)
  for (long inn = 0; inn < N; inn++)
    for (long knn = 0; knn < K; knn++)
      for (int inc = 0; inc < C ; inc += FP32_PER_REG)
        for (long outh = (outHeight / TILE_OUT_HW) * TILE_OUT_HW; outh < outHeight; outh++) {
          for (long outw = 0; outw < (outWidth / TILE_OUT_HW) * TILE_OUT_HW; outw++) {
            outpos = inn * outgap[0] + knn * outgap[1] + outh * outgap[2] + outw;
            svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, C - inc));

            for (long knh = 0; knh < 3; knh++) {
              for (long knw = 0; knw < 3; knw++) {
                inpos = inn * ingap[0] + (outh + knh) * W * C + (outw + knw) * C + inc;
                knpos = knn * kngap[0] + knh * 3 * C + knw * C + inc;

                svfloat32_t z0 = svld1(pg, packedImage + inpos);
                svfloat32_t z1 = svld1(pg, packedFilter + knpos);

                svfloat32_t z3 = svmul_f32_x(pg, z0, z1);
                float sum = svaddv_f32(pg, z3);
                out[outpos] += sum;
              }
            }
          }
        }
  }

  if (outWidth % TILE_OUT_HW > 0)
  #pragma omp parallel for private(inpos, knpos, outpos)
  for (long inn = 0; inn < N; inn++)
    for (long knn = 0; knn < K; knn++)
      for (int inc = 0; inc < C ; inc += FP32_PER_REG) {
        for(long outh = 0; outh < (outHeight / TILE_OUT_HW) * TILE_OUT_HW; outh++){
          for (long outw = (outWidth / TILE_OUT_HW) * TILE_OUT_HW; outw < outWidth; outw++){
            outpos = inn * outgap[0] + knn * outgap[1] + outh * outgap[2] + outw;
            svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, C - inc));

            for (long knh = 0; knh < 3; knh++) {
              for (long knw = 0; knw < 3; knw++) {
                inpos = inn * ingap[0] + (outh + knh) * W * C + (outw + knw) * C + inc;
                knpos = knn * kngap[0] + knh * 3 * C + knw * C + inc;

                svfloat32_t z0 = svld1(pg, packedImage + inpos);
                svfloat32_t z1 = svld1(pg, packedFilter + knpos);

                svfloat32_t z3 = svmul_f32_x(pg, z0, z1);
                float sum = svaddv_f32(pg, z3);
                out[outpos] += sum;
              }
            }
          }
        }
      }

  if(outHeight % TILE_OUT_HW > 0 && outWidth % TILE_OUT_HW > 0)
  #pragma omp parallel for private(inpos, knpos, outpos)
  for (long inn = 0; inn < N; inn++)
    for (long knn = 0; knn < K; knn++)
      for (int inc = 0; inc < C ; inc += FP32_PER_REG) 
        for(long outh = (outHeight / TILE_OUT_HW) * TILE_OUT_HW; outh < outHeight; outh++)
          for (long outw = (outWidth / TILE_OUT_HW) * TILE_OUT_HW; outw < outWidth; outw++){
            outpos = inn * outgap[0] + knn * outgap[1] + outh * outgap[2] + outw;
            svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, C - inc));

            for (long knh = 0; knh < 3; knh++) {
              for (long knw = 0; knw < 3; knw++) {
                inpos = inn * ingap[0] + (outh + knh) * W * C + (outw + knw) * C + inc;
                knpos = knn * kngap[0] + knh * 3 * C + knw * C + inc;

                svfloat32_t z0 = svld1(pg, packedImage + inpos);
                svfloat32_t z1 = svld1(pg, packedFilter + knpos);

                svfloat32_t z3 = svmul_f32_x(pg, z0, z1);
                float sum = svaddv_f32(pg, z3);
                out[outpos] += sum;
              }
            }
          }

}

ALWAYS_INLINE void filterIcPack(float* __restrict__ filter, FltShape fs, float* __restrict__ packedFiler) {
  int K = fs.oc, C = fs.ic;
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)            // 注意「伪共享」
  for(int k = 0; k < K; ++k)
    for(int h = 0; h < FLT_HW; ++h)
      for(int w = 0; w < FLT_HW; ++w)
        for(int c = 0; c < C; ++c)
          packedFiler[k * 3 * 3 * C + h * 3 * C + w * C + c] 
                    = filter[k * C * 3 * 3 + c * 3 * 3 + h * 3 + w];
}

ALWAYS_INLINE void ImageIcPack(float* __restrict__ Image, ImgShape is,  float* __restrict__ packedImage) {
  int N = is.numImg, C = is.ic, H = is.h, W = is.w;
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for(int n = 0; n < N; ++n)
    for(int h = 0; h < H; ++h)
      for(int w = 0; w < W; ++w)
        for(int c = 0; c < C; ++c)
          packedImage[n * H * W * C + h * W * C + w * C + c] 
                    = Image[n * C * H * W + c * H * W + h * W + w];
}

// trancform the (k, c) filter from filer_KHWC, store to u_arr[K][6][6][C]
ALWAYS_INLINE void filterTransformSVE(float* __restrict__ packedFiler, int C, float* __restrict__ U, int k, int c) {
  float* filter_ptr = packedFiler + k * 3 * 3 * C;
  float* u_arr_ptr = U + k * 6 * 6 * C;
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

ALWAYS_INLINE void filterTransform(float* __restrict__ packedFilter, FltShape fs, float* __restrict__ U) {
  int K = fs.oc, C = fs.ic;
  PRAGMA_OMP_PARALLEL_FOR()
  for (int k = 0; k < K; ++k) {
    for(int c = 0; c < C; c += FP32_PER_REG) {
      filterTransformSVE(packedFilter, C, U, k, c);
    }
  }
}

ALWAYS_INLINE void srcTransformSVE(float* __restrict__ packedImage, ImgShape is,  float* V, VShape vs, int tileNo, TileShape ts, int c) {
  TileIndex ti = getTileIndex(tileNo, ts);
  int n = ti.b, x = ti.tw, y = ti.th;
  int inHeight = is.h, inWidth = is.w, C = is.ic;
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
  float tmp[TILE_IN_H][TILE_IN_W][FP32_PER_REG] ATTRIBUTE_ALIGN(128);
  svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, C-c));

  float (*Varr) [TILE_IN_W][C] = (float (*)[TILE_IN_W][C]) (V + tileNo * TILE_IN_H * TILE_IN_W * C);

  for (int xx = 0; xx < TILE_IN_W; ++xx) {   // 按列产生结果。
    z6 = svld1(pg, packedImage + n * inHeight * inWidth * C + (y * 4 + 0) * inWidth * C + (x * 4 + xx) * C + c);

    z0 = svmul_f32_x(pg, z28, z6);

    z6 = svld1(pg, packedImage + n * inHeight * inWidth * C + (y * 4 + 1) * inWidth * C + (x * 4 + xx) * C + c);

    z1 = svmul_f32_x(pg, z29, z6);
    z2 = svmul_f32_x(pg, z28, z6);
    z3 = svmul_f32_x(pg, z27, z6);
    z4 = svmul_f32_x(pg, z26, z6);
    z5 = svmul_f32_x(pg, z28, z6);

    z6 = svld1(pg, packedImage + n * inHeight * inWidth * C + (y * 4 + 2) * inWidth * C + (x * 4 + xx) * C + c);

    z0 = svmla_f32_x(pg, z0, z31, z6);
    z1 = svmla_f32_x(pg, z1, z29, z6);
    z2 = svmla_f32_x(pg, z2, z29, z6);
    z3 = svmla_f32_x(pg, z3, z25, z6);
    z4 = svmla_f32_x(pg, z4, z25, z6);

    z6 = svld1(pg, packedImage + n * inHeight * inWidth * C + (y * 4 + 3) * inWidth * C + (x * 4 + xx) * C + c);

    z1 = svmla_f32_x(pg, z1, z24, z6);
    z2 = svmla_f32_x(pg, z2, z25, z6);
    z3 = svmla_f32_x(pg, z3, z26, z6);
    z4 = svmla_f32_x(pg, z4, z27, z6);
    z5 = svmla_f32_x(pg, z5, z31, z6);

    z6 = svld1(pg, packedImage + n * inHeight * inWidth * C + (y * 4 + 4) * inWidth * C + (x * 4 + xx) * C + c);
    
    z0 = svmla_f32_x(pg, z0, z24, z6);
    z1 = svmla_f32_x(pg, z1, z24, z6);
    z2 = svmla_f32_x(pg, z2, z24, z6);
    z3 = svmla_f32_x(pg, z3, z24, z6);
    z4 = svmla_f32_x(pg, z4, z24, z6);

    z6 = svld1(pg, packedImage + n * inHeight * inWidth * C + (y * 4 + 5) * inWidth * C + (x * 4 + xx) * C + c);

    z5 = svmla_f32_x(pg, z5, z24, z6);

    svst1_f32(pg, tmp[0][xx], z0);
    svst1_f32(pg, tmp[1][xx], z1);
    svst1_f32(pg, tmp[2][xx], z2);
    svst1_f32(pg, tmp[3][xx], z3);
    svst1_f32(pg, tmp[4][xx], z4);
    svst1_f32(pg, tmp[5][xx], z5);
  }

  for (int yy = 0; yy < TILE_IN_H; ++yy) {   // 按行产生结果。
    z6 = svld1(pg, tmp[yy][0]);

    z0 = svmul_f32_x(pg, z28, z6);

    z6 = svld1(pg, tmp[yy][1]);

    z1 = svmul_f32_x(pg, z29, z6);
    z2 = svmul_f32_x(pg, z28, z6);
    z3 = svmul_f32_x(pg, z27, z6);
    z4 = svmul_f32_x(pg, z26, z6);
    z5 = svmul_f32_x(pg, z28, z6);

    z6 = svld1(pg, tmp[yy][2]);

    z0 = svmla_f32_x(pg, z0, z31, z6);
    z1 = svmla_f32_x(pg, z1, z29, z6);
    z2 = svmla_f32_x(pg, z2, z29, z6);
    z3 = svmla_f32_x(pg, z3, z25, z6);
    z4 = svmla_f32_x(pg, z4, z25, z6);

    z6 = svld1(pg, tmp[yy][3]);

    z1 = svmla_f32_x(pg, z1, z24, z6);
    z2 = svmla_f32_x(pg, z2, z25, z6);
    z3 = svmla_f32_x(pg, z3, z26, z6);
    z4 = svmla_f32_x(pg, z4, z27, z6);
    z5 = svmla_f32_x(pg, z5, z31, z6);

    z6 = svld1(pg, tmp[yy][4]);
    
    z0 = svmla_f32_x(pg, z0, z24, z6);
    z1 = svmla_f32_x(pg, z1, z24, z6);
    z2 = svmla_f32_x(pg, z2, z24, z6);
    z3 = svmla_f32_x(pg, z3, z24, z6);
    z4 = svmla_f32_x(pg, z4, z24, z6);

    z6 = svld1(pg, tmp[yy][5]);

    z5 = svmla_f32_x(pg, z5, z24, z6);

    svst1_f32(pg, &Varr[yy][0][c], z0);
    svst1_f32(pg, &Varr[yy][1][c], z1);
    svst1_f32(pg, &Varr[yy][2][c], z2);
    svst1_f32(pg, &Varr[yy][3][c], z3);
    svst1_f32(pg, &Varr[yy][4][c], z4);
    svst1_f32(pg, &Varr[yy][5][c], z5);
  }
}

ALWAYS_INLINE void copyFilter(float* __restrict__ U, int C, int k, int c, float u[TILE_IN_H][TILE_IN_W][FP32_PER_REG]) {
  for(int yy = 0; yy < TILE_IN_H; ++yy) {
    for(int xx = 0; xx < TILE_IN_W; ++xx) {
      float* pos = U + k * TILE_IN_H * TILE_IN_W * C + yy * TILE_IN_W * C + xx * C + c;
      svst1_f32(svptrue_b32(), &u[yy][xx][0], svld1(svptrue_b32(), pos));
    }
  }
}

ALWAYS_INLINE void destTransformSVE(float M[TILE_IN_H][TILE_IN_W][FP32_PER_REG], int K, int kk, float Y[TILE_OUT_H][TILE_IN_W][FP32_PER_REG]) {
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

ALWAYS_INLINE void destStore(float Y[TILE_OUT_H][TILE_IN_W][FP32_PER_REG], float* __restrict__ out, OutShape os, int tileNo, TileShape ts, int kk) {
  int K = os.oc, outHeight = os.h, outWidth = os.w;
  TileIndex ti = getTileIndex(tileNo, ts);
  int n = ti.b, x = ti.tw, y = ti.th;
  for(int k = kk; k < kk + MIN(FP32_PER_REG, K - kk); ++k)
    for(int yy = 0; yy < TILE_OUT_H; ++yy)
      for(int xx = 0; xx < TILE_OUT_W; ++xx)
        out[(long)((n * K + k) * outHeight + y * 4 + yy) * outWidth + x * 4 + xx] = Y[yy][xx][k - kk];
}

ALWAYS_INLINE void hadamardProduct(float U[TILE_IN_H][TILE_IN_W][FP32_PER_REG], float* V, VShape vs, int tileNo, int kk, int k,int C, int c, float M[TILE_IN_H][TILE_IN_W][FP32_PER_REG]) {
  for(int yy = 0; yy < TILE_IN_H; ++yy) {
    for(int xx = 0; xx < TILE_IN_W; ++xx) {
      DECLARE_SVE_FP32_REGS()
      svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, C-c));
      z0 = svld1(pg, V + tileNo * TILE_IN_H * TILE_IN_W * C + yy * TILE_IN_W * C + xx * C + c);
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
                 float *__restrict__ U_no_use, float *__restrict__ V_no_use,
                 float *__restrict__ M) {

  /* new vars of shape */
  ImgShape is = {N, C, inHeight, inWidth};
  FltShape fs = {K, C, FLT_H, FLT_W};
  OutShape os = getOutShape(is, fs);
  TileShape ts = getTileShape(is, os);
  UShape us = getUShape(fs);
  VShape vs = getVShape(is, ts);

  int outgap[3] = {os.oc * os.h * os.w,  os.h * os.w,  os.w};
  #pragma omp parallel for simd aligned(out) schedule(static)
  for(int i = 0; i < N * outgap[0]; ++i) out[i] = 0;

  float* filerPacked =  (float*)  aligned_alloc(ALLOC_ALIGNMENT, K * C * 3 * 3 * sizeof(float));
  assert(filerPacked != NULL);
  float* imagePacked =  (float*)  aligned_alloc(ALLOC_ALIGNMENT, sizeof(float) * N * C * inHeight * inWidth);
  assert(imagePacked != NULL);
  float* V = (float*)  aligned_alloc(ALLOC_ALIGNMENT, sizeof(float) * vs.numTileTotal  * 6 * 6 * C);
  assert(V != NULL);
  float* U = (float*)  aligned_alloc(ALLOC_ALIGNMENT, sizeof(float) * K * 6 * 6 * C);
  assert(U != NULL);

  filterIcPack(filter, fs, filerPacked);
  filterTransform(filerPacked, fs, U);
  ImageIcPack(image, is, imagePacked);

  PRAGMA_OMP_PARALLEL_FOR()
  for (int tileNo = 0; tileNo < vs.numTileTotal; ++tileNo) {
    for (int c = 0; c < C; c += FP32_PER_REG) {
      srcTransformSVE(imagePacked, is, V, vs, tileNo, ts, c);
    }
  }
  
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for (int kk = 0; kk < K; kk += FP32_PER_REG) {
    for (int tileNo = 0; tileNo < vs.numTileTotal; ++tileNo) {

      float M[6][6][FP32_PER_REG] ATTRIBUTE_ALIGN(128);
      memset(M, 0, sizeof M);
      float Y[4][6][FP32_PER_REG]   ATTRIBUTE_ALIGN(128);

      for(int k = kk; k < kk + MIN(FP32_PER_REG, K - kk); ++k) {
        for (int c = 0; c < C; c += FP32_PER_REG) {
          float u[TILE_IN_H][TILE_IN_W][FP32_PER_REG] ATTRIBUTE_ALIGN(128);
          copyFilter(U, C, k, c, u);
          hadamardProduct(u, V, vs, tileNo, kk, k, C, c, M);
        }
      }

      destTransformSVE(M, K, kk, Y);
      destStore(Y, out, os, tileNo, ts, kk);
    }
  }


  // 边界处理
  boundaryTreatment(imagePacked, is,  filerPacked, fs, out);

  free(U);
  free(imagePacked);
  free(filerPacked);
}