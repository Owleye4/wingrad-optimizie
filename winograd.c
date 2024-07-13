#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  for (int i = 0; i < 16; ++i){
    out[i] = 0.0f;
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 6; ++k)
        out[i * 4 + j] += A[i * 6 + k] * B[k * 4 + j];
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

  float* u_arr = (float*) malloc(K * C * sizeof(float*) * 36);
  assert(u_arr != NULL);
  #pragma omp parallel for
  for (int k = 0; k < K; ++k){
    for (int c = 0; c < C; ++c){
      float tmp_u[18]; // 6 * 3 
      float* filters_ptr = filter + (k * C + c) * sizeF;
      float* u = u_arr + (k * C + c) * 36;
      sgemm_6x3x3(&G[0][0], filters_ptr, tmp_u, 6, 3, 3);
      sgemm_6x3x6(tmp_u, &G_T[0][0], u, 6, 3, 6);
    }
  }
  // U[:, :, k, c] = G * filters[k, c, :, :] * G.T()
  #pragma omp parallel for collapse(3)// private(tmp_u, u)
  // #pragma omp parallel for collapse(2)// private(tmp_v, d, v)
  for (int n = 0; n < N; ++n) {
    for (int y = 0; y < outHeight / 4; ++y) {
      for (int x = 0; x < outWidth / 4; ++x) {
        for (int c = 0; c < C; ++c) {
          for (int k = 0; k < K; ++k) {
            float tmp_v[36];
            float M[36];
            float d[36];  // d: [6 * 6];
            float v[36];  // v: [6 * 6];
            float *u = u_arr + (k * C + c) * 36;
            // Generate d_cb
            for (int iy = 0; iy < 6; ++iy) 
              for (int ix = 0; ix < 6; ++ix) 
                d[iy * 6 + ix] = image[(n * C + c) * sizeI +
                                      (y * 4 + iy) * inWidth + (x * 4 + ix)];
            sgemm_6x6x6(&B_T[0][0], d, tmp_v, 6, 6, 6);
            sgemm_6x6x6(tmp_v, &B[0][0], v, 6, 6, 6);
            // int b = ((n * outHeight / 2) + y) * outWidth / 2 + x;
            for(int i = 0; i < 36; ++i) 
              M[i] = u[i] * v[i];
            float A_TxM[24]; //  A_TxM: [4 * 6];
            float Tile[16];  //  Tile: [4 * 4]
            sgemm_4x6x6(&A_T[0][0], &M[0], &A_TxM[0], 4, 6, 6);
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
  free(u_arr);
}