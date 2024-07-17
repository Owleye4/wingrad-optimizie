#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_sve.h>
#include "common.h"


const float G[6][3] = {
    {1.0 / 4,      0.0,       0.0},
    {-1.0 / 6, -1.0 / 6, -1.0 / 6},
    {-1.0 / 6,  1.0 / 6, -1.0 / 6},
    {1.0 / 24,  1.0 / 12, 1.0 / 6},
    {1.0 / 24, -1.0 / 12, 1.0 / 6},
    {0.0,            0.0,     1.0}};

const float G_T[3][6] = {
    {1.0 / 4, -1.0 / 6, -1.0 / 6, 1.0 / 24,  1.0 / 24, 0.0},
    {    0.0, -1.0 / 6,  1.0 / 6, 1.0 / 12, -1.0 / 12, 0.0},
    {    0.0, -1.0 / 6, -1.0 / 6,  1.0 / 6,   1.0 / 6, 1.0}};

const float B[6][6] = {
    { 4.0,  0.0,  0.0,  0.0,  0.0, 0.0},
    { 0.0, -4.0,  4.0, -2.0,  2.0, 4.0},
    {-5.0, -4.0, -4.0, -1.0, -1.0, 0.0},
    { 0.0,  1.0, -1.0,  2.0, -2.0,-5.0},
    { 1.0,  1.0,  1.0,  1.0,  1.0, 0.0},
    { 0.0,  0.0,  0.0,  0.0,  0.0, 1.0}};

const float B_T[6][6] = {
    {4.0,  0.0, -5.0,  0.0,  1.0, 0.0},
    {0.0, -4.0, -4.0,  1.0,  1.0, 0.0},
    {0.0,  4.0, -4.0, -1.0,  1.0, 0.0},
    {0.0, -2.0, -1.0,  2.0,  1.0, 0.0},
    {0.0,  2.0, -1.0, -2.0,  1.0, 0.0},
    {0.0,  4.0,  0.0, -5.0,  0.0, 1.0}};

const float A[6][4] = {
    {1.0,  0.0, 0.0,  0.0},
    {1.0,  1.0, 1.0,  1.0},
    {1.0, -1.0, 1.0, -1.0},
    {1.0,  2.0, 4.0,  8.0},
    {1.0, -2.0, 4.0, -8.0},
    {0.0,  0.0, 0.0,  1.0}};

const float A_T[4][6] = {
    {1.0, 1.0,  1.0, 1.0, 1.0, 0.0},
    {0.0, 1.0, -1.0, 2.0,-2.0, 0.0},
    {0.0, 1.0,  1.0, 4.0, 4.0, 0.0},
    {0.0, 1.0, -1.0, 8.0,-8.0, 1.0}};

void sgemm_6x3x3(const float *A, const float *B, float *out, const int M, const int K, const int N)
{
  for (int i = 0; i < 18; ++i){
    out[i] = 0.0f;
  }
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        out[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
}

void sgemm_6x3x6(const float *A, const float *B, float *out, const int M, const int K, const int N)
{
  for (int i = 0; i < 36; ++i){
    out[i] = 0.0f;
  }
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      for (int k = 0; k < 3; ++k)
        out[i * 6 + j] += A[i * 3 + k] * B[k * 6 + j];
}

void sgemm_6x6x6(const float *A, const float *B, float *out, const int M, const int K, const int N)
{
  for (int i = 0; i < 36; ++i){
    out[i] = 0.0f;
  }
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      for (int k = 0; k < 6; ++k)
        out[i * 6 + j] += A[i * 6 + k] * B[k * 6 + j];
}

void sgemm_4x6x6(const float *A, const float *B, float *out, const int M, const int K, const int N)
{
  for (int i = 0; i < 24; ++i){
    out[i] = 0.0f;
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 6; ++j)
      for (int k = 0; k < 6; ++k)
        out[i * 6 + j] += A[i * 6 + k] * B[k * 6 + j];
}

void sgemm_4x6x4(const float *A, const float *B, float *out, const int M, const int K, const int N)
{
  for (int i = 0; i < 16; ++i) {
    out[i] = 0.0f;
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 6; ++k)
        out[i * 4 + j] += A[i * 6 + k] * B[k * 4 + j];
}

void filter_KCHW_to_KHWC(float* __restrict__ filter, int K, int C, float* __restrict__ filer_KHWC) {
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for(int k = 0; k < K; ++k)
    for(int h = 0; h < FLT_HW; ++h)
      for(int w = 0; w < FLT_HW; ++w) {
        for(int c = 0; c < C; ++c)
          // 注意伪共享
          filer_KHWC[k * 3 * 3 * C + h * 3 * C + w * C + c] 
                    = filter[k * C * 3 * 3 + c * 3 * 3 + h * 3 + w];
        }
}

void u_arr_K66C_to_KC66(float* __restrict__ u_arr, int K, int C, float* __restrict__ u_arr_out) {
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for(int k = 0; k < K; ++k)
    for(int h = 0; h < TILE_IN_HW; ++h)
      for(int w = 0; w < TILE_IN_HW; ++w) {
        for(int c = 0; c < C; ++c)
          u_arr_out[k * C * 6 * 6 + c * 6 * 6 + h * 6 + w] 
                    = u_arr[k * 6 * 6 * C + h * 6 * C + w * C + c];
        }
}


void Image_NCHW_to_NHWC(float* __restrict__ Image, int N, int C, int H, int W, float* __restrict__ Image_NHWC) {
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for(int n = 0; n < N; ++n)
    for(int h = 0; h < H; ++h)
      for(int w = 0; w < W; ++w) {
        for(int c = 0; c < C; ++c)
          Image_NHWC[n * H * W * C + h * W * C + w * C + c] 
                    = Image[n * C * H * W + c * H * W + h * W + w];
        }
}



void filter_transform(float* __restrict__ filer_KHWC, int K,  int C, float* __restrict__ u_arr) {
  // filer_KHWC[K][3][3][C]
  // u_arr[K][6][6][C]
  float* tmp_u = malloc(sizeof(float) * K * 6 * 3 * C); 
  PRAGMA_OMP_PARALLEL_FOR()
  for (int k = 0; k < K; ++k) {
    float* filter_ptr = filer_KHWC + k * 3 * 3 * C;
    float* u_arr_ptr = u_arr + k * 6 * 6 * C;
    float* tmp_u_ptr = tmp_u + k * 6 * 3 * C;
    DECLARE_SVE_FP32_REGS();
    z25 = svdup_f32(1.0f);
    z26 = svdup_f32( -1.0f / 6.0f  );
    z27 = svdup_f32( -1.0f / 12.0f );
    z28 = svdup_f32(  1.0f / 4.0f  );
    z29 = svdup_f32(  1.0f / 6.0f  );
    z30 = svdup_f32(  1.0f / 12.0f );
    z31 = svdup_f32(  1.0f / 24.0f  );
    for(int c = 0; c < C; c += FP32_PER_REG) {
      svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, C-c));
      // G * filter
      for (int i = 0; i < FLT_HW; ++i) {     // 这个循环按row遍历filter（按G以及结果的column）， 按列产生结果。
        z6 = svld1(pg, filter_ptr + i * 3 * C + 0 * C + c);
        
        z0 = svmul_f32_x(pg, z28, z6);
        z1 = svmul_f32_x(pg, z26, z6);
        z2 = svmul_f32_x(pg, z26, z6);
        z3 = svmul_f32_x(pg, z31, z6);
        z4 = svmul_f32_x(pg, z31, z6);

        z6 = svld1(pg, filter_ptr + i * 3 * C + 1 * C + c);
        
        // z0 += 0;
        z1 = svmla_f32_x(pg, z1, z26, z6);
        z2 = svmla_f32_x(pg, z2, z29, z6);
        z3 = svmla_f32_x(pg, z3, z30, z6);
        z4 = svmla_f32_x(pg, z4, z27, z6);
        // z5 += 0;
        z6 = svld1(pg, filter_ptr + i * 3 * C + 2 * C + c);
        
        // z0 += 0;
        z1 = svmla_f32_x(pg, z1, z26, z6);
        z2 = svmla_f32_x(pg, z2, z26, z6);
        z3 = svmla_f32_x(pg, z3, z29, z6);
        z4 = svmla_f32_x(pg, z4, z29, z6);
        z5 = z6;

        svst1_f32(pg, tmp_u_ptr + 0 * 3 * C + i * C + c, z0);
        svst1_f32(pg, tmp_u_ptr + 1 * 3 * C + i * C + c, z1);
        svst1_f32(pg, tmp_u_ptr + 2 * 3 * C + i * C + c, z2);
        svst1_f32(pg, tmp_u_ptr + 3 * 3 * C + i * C + c, z3);
        svst1_f32(pg, tmp_u_ptr + 4 * 3 * C + i * C + c, z4);
        svst1_f32(pg, tmp_u_ptr + 5 * 3 * C + i * C + c, z5);
      }
      // (G * filter) * G_T
      for (int i = 0; i < TILE_IN_HW; ++i) {    // 这个循环按row遍历(G * filter)（按G_T的column遍历），按行产生结果。
        z6 = svld1(pg, tmp_u_ptr + i * 3 * C + 0 * C + c);
        
        z0 = svmul_f32_x(pg, z28, z6);
        z1 = svmul_f32_x(pg, z26, z6);
        z2 = svmul_f32_x(pg, z26, z6);
        z3 = svmul_f32_x(pg, z31, z6);
        z4 = svmul_f32_x(pg, z31, z6);

        z6 = svld1(pg, tmp_u_ptr + i * 3 * C + 1 * C + c);
        
        // z0 += 0;
        z1 = svmla_f32_x(pg, z1, z26, z6);
        z2 = svmla_f32_x(pg, z2, z29, z6);
        z3 = svmla_f32_x(pg, z3, z30, z6);
        z4 = svmla_f32_x(pg, z4, z27, z6);
        // z5 += 0;
        z6 = svld1(pg, tmp_u_ptr + i * 3 * C + 2 * C + c);
        
        // z0 += 0;
        z1 = svmla_f32_x(pg, z1, z26, z6);
        z2 = svmla_f32_x(pg, z2, z26, z6);
        z3 = svmla_f32_x(pg, z3, z29, z6);
        z4 = svmla_f32_x(pg, z4, z29, z6);
        z5 = z6;

        svst1_f32(pg, u_arr_ptr + 0 * 6 * C + i * C + c, z0);
        svst1_f32(pg, u_arr_ptr + 1 * 6 * C + i * C + c, z1);
        svst1_f32(pg, u_arr_ptr + 2 * 6 * C + i * C + c, z2);
        svst1_f32(pg, u_arr_ptr + 3 * 6 * C + i * C + c, z3);
        svst1_f32(pg, u_arr_ptr + 4 * 6 * C + i * C + c, z4);
        svst1_f32(pg, u_arr_ptr + 5 * 6 * C + i * C + c, z5);
      }
    }
  }
  free(tmp_u);
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

  // m = 2; r = 3; alpha = 4
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
  memset(out, 0, N * outgap[0] * sizeof(float));

  float* u_arr = (float*) malloc(sizeof(float) * K * 6 * 6 * C);
  float* u_arr_trans = (float*) malloc(sizeof(float) * K * 6 * 6 * C);
  assert(u_arr != NULL);
  assert(u_arr_trans != NULL);
  float* filer_KHWC = (float*) malloc(K * C * 3 * 3 * sizeof(float));
  assert(filer_KHWC != NULL);

  filter_KCHW_to_KHWC(filter, K, C, filer_KHWC);
  filter_transform(filer_KHWC, K, C, u_arr);

  float* image_NHWC =  (float*) malloc(sizeof(float) * N * C * inHeight * inWidth);

  Image_NCHW_to_NHWC(image, N, C, inHeight, inWidth, image_NHWC);

  // u_arr_K66C_to_KC66(u_arr, K, C, u_arr_trans);

  free(u_arr);
  u_arr = u_arr_trans;

  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      for (int y = 0; y < outHeight / 4; ++y) {
        for (int x = 0; x < outWidth / 4; ++x) {
          DECLARE_SVE_FP32_REGS();
          z24 = svdup_f32(  1.0f );
          z25 = svdup_f32( -1.0f );
          z26 = svdup_f32(  2.0f );
          z27 = svdup_f32( -2.0f );
          z28 = svdup_f32(  4.0f );
          z29 = svdup_f32( -4.0f );
          z30 = svdup_f32(  5.0f );
          z31 = svdup_f32( -5.0f );
          for (int c = 0; c < C; c += FP32_PER_REG) {
            /*
              // float tmp_v[36];
              // float M[36];
              // float d[36];  // d: [6 * 6];
              // float v[36];  // v: [6 * 6];
              // float *u = u_arr + (k * C + c) * 36;
              // // Generate d_cb
              // for (int iy = 0; iy < 6; ++iy) 
              //   for (int ix = 0; ix < 6; ++ix) 
              //     d[iy * 6 + ix] = image[(n * C + c) * sizeI +
              //                           (y * 4 + iy) * inWidth + (x * 4 + ix)];
              // sgemm_6x6x6(&B_T[0][0], d, tmp_v, 6, 6, 6);
              // sgemm_6x6x6(tmp_v, &B[0][0], v, 6, 6, 6);
              // // int b = ((n * outHeight / 2) + y) * outWidth / 2 + x;
              // for(int i = 0; i < 36; ++i) 
              //   M[i] = u[i] * v[i];
              // float A_TxM[24]; //  A_TxM: [4 * 6];
              // float Tile[16];  //  Tile: [4 * 4]
              // sgemm_4x6x6(&A_T[0][0], &M[0], &A_TxM[0], 4, 6, 6);
              // sgemm_4x6x4(&A_TxM[0], &A[0][0], &Tile[0], 4, 6, 4);
              // for(int outh=0; outh < 4; ++outh)
              //   for(int outw=0; outw < 4; ++outw) {
              //     int outpos = n * outgap[0] + k * outgap[1] + (y * 4 + outh) * outgap[2] + (x * 4 + outw); 
              //     out[outpos] += Tile[outh * 4 + outw];
              //   } 
              // ==============================================================================================
            */
            svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, C-c));
            // B_T * d
            float U[6][6][FP32_PER_REG];
            float V[6][6][FP32_PER_REG];
            float M[6][6];
            for(int yy = 0; yy < TILE_IN_HW; ++yy) 
              for(int xx = 0; xx < TILE_IN_HW; ++xx){
                svst1_f32(svptrue_b32(), (float *)&U[yy][xx], svld1(svptrue_b32(), u_arr + k * FLT_HW * FLT_HW * C + yy * FLT_HW * C + xx * C + c));
              }


            float tmp[6][6][FP32_PER_REG];
            for (int xx = 0; xx < TILE_IN_HW; ++xx) {   // 按列产生结果。
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

              svst1_f32(pg, tmp[0][xx], z0);
              svst1_f32(pg, tmp[1][xx], z1);
              svst1_f32(pg, tmp[2][xx], z2);
              svst1_f32(pg, tmp[3][xx], z3);
              svst1_f32(pg, tmp[4][xx], z4);
              svst1_f32(pg, tmp[5][xx], z5);
            }

            for (int yy = 0; yy < TILE_IN_HW; ++yy) {   // 按列产生结果。
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

              svst1_f32(pg, V[yy][0], z0);
              svst1_f32(pg, V[yy][1], z1);
              svst1_f32(pg, V[yy][2], z2);
              svst1_f32(pg, V[yy][3], z3);
              svst1_f32(pg, V[yy][4], z4);
              svst1_f32(pg, V[yy][5], z5);
            }
            for(int xx = 0; xx < TILE_IN_HW; ++xx)
              for(int yy = 0; yy < TILE_IN_HW; ++yy) {
                z0 = svld1(pg, &V[yy][xx][0]);
                z1 = svld1(pg, &U[yy][xx][0]);
                svfloat32_t UV = svmul_f32_x(pg, z0, z1);
                M[yy][xx] = svaddv_f32(pg, UV); //reduce
              }
            
            float Tile[16]; 
            float A_TxM[24];
            sgemm_4x6x6(&A_T[0][0], &M[0][0], &A_TxM[0], 4, 6, 6);
            sgemm_4x6x4(&A_TxM[0], &A[0][0], &Tile[0], 4, 6, 4);
            for(int outh=0; outh < 4; ++outh)
              for(int outw=0; outw < 4; ++outw) {
                int outpos = n * outgap[0] + k * outgap[1] + (y * 4 + outh) * outgap[2] + (x * 4 + outw); 
                out[outpos] += Tile[outh * 4 + outw];
              }
          }
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