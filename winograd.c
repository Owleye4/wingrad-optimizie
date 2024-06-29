#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const float G[4][3] = {
    {1.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0.0, 0.0, 1.0}};
const float G_T[3][4] = {
    {1, 0.5, 0.5, 0.0}, {0.0, 0.5, -0.5, 0.0}, {0.0, 0.5, 0.5, 1.0}};
const float B[4][4] = {
    {1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};
const float B_T[4][4] = {
    {1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
const float A[4][2] = {{1, 0}, {1, 1}, {1, -1}, {0, -1}};
const float A_T[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};

// Matrix Multiplication: Out = A x B (A:M*K, B:K*N, out: M*N)
// All arrays should have their memory prepared correctly outside this function
// For rookies: this sgemm is the worst sgemm I've ever written throughout my
// career.
//      If you don't know where to start, optimize this function as a good
//      starting point.
void sgemm_4x3x3(const float *A, const float *B, float *out, const int M, const int K,
           const int N) {
  for (int i = 0; i < 12; ++i) {
    out[i] = 0.0f;
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
       for (int k = 0; k < 3; ++k)
          out[i * 3 + j]  += A[i * 3 + k] * B[k * 3 + j];
}

void sgemm_4x3x4(const float *A, const float *B, float *out, const int M, const int K,
           const int N) {
  for (int i = 0; i < 16; ++i) {
    out[i] = 0.0f;
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
       for (int k = 0; k < 3; ++k)
          out[i * 4 + j]  += A[i * 3 + k] * B[k * 4 + j];
}

void sgemm_2x4x4(const float *A, const float *B, float *out, const int M, const int K,
           const int N) {
  for (int i = 0; i < 8; ++i) {
    out[i] = 0.0f;
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
       for (int k = 0; k < 4; ++k)
          out[i * 4 + j]  += A[i * 4 + k] * B[k * 4 + j];
}

void sgemm_2x4x2(const float *A, const float *B, float *out, const int M, const int K,
           const int N) {
  for (int i = 0; i < 4; ++i) {
    out[i] = 0.0f;
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
       for (int k = 0; k < 4; ++k)
          out[i * 2 + j]  += A[i * 4 + k] * B[k * 2 + j];
}

void sgemm_4x4x4(const float *A, const float *B, float *out, const int M, const int K,
           const int N) {
  for (int i = 0; i < 16; ++i) {
    out[i] = 0.0f;
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
       for (int k = 0; k < 4; ++k)
          out[i * 4 + j]  += A[i * 4 + k] * B[k * 4 + j];
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
  const long P = outHeight / 2 * outWidth / 2 * N;
  int outgap[3] = {
      K * (inHeight - 2) * (inWidth - 2), 
      (inHeight - 2) * (inWidth - 2), 
      (inWidth - 2)
    };
  memset(out, 0, N * outgap[0]);

  // U[:, :, k, c] = G * filters[k, c, :, :] * G.T()
  #pragma omp parallel for// private(tmp_u, u)
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      float tmp_u[12];  // 4 * 3
      float u[16];      // 4 * 4;
      float *filters_ptr = filter + (k * C + c) * sizeF;
      sgemm_4x3x3(&G[0][0], filters_ptr, tmp_u, 4, 3, 3);
      sgemm_4x3x4(tmp_u, &G_T[0][0], u, 4, 3, 4);
      #pragma omp parallel for collapse(2)// private(tmp_v, d, v)
      for (int n = 0; n < N; ++n) {
        for (int y = 0; y < outHeight / 2; ++y) {
          for (int x = 0; x < outWidth / 2; ++x) {
            float tmp_v[16];
            float M[16];
            float d[16];  // d: [4 * 4];
            float v[16];  // v: [4 * 4];
            // Generate d_cb
            for (int iy = 0; iy < 4; ++iy) 
              for (int ix = 0; ix < 4; ++ix) 
                d[iy * 4 + ix] = image[(n * C + c) * sizeI +
                                      (y * 2 + iy) * inWidth + (x * 2 + ix)];
            sgemm_4x4x4(&B_T[0][0], d, tmp_v, 4, 4, 4);
            sgemm_4x4x4(tmp_v, &B[0][0], v, 4, 4, 4);
            int b = ((n * outHeight / 2) + y) * outWidth / 2 + x;
            for(int i = 0; i < 16; ++i) 
              M[i] = u[i] * v[i];
            float A_TxM[8]; //size{A_TxM} = (2,4)
            float Tile[4];  //size{Tile} = (2,2)
            sgemm_2x4x4(&A_T[0][0], &M[0], &A_TxM[0], 2, 4, 4);
            sgemm_2x4x2(&A_TxM[0], &A[0][0], &Tile[0], 2, 4, 2);
            for(int outh=0; outh < 2; ++outh)
              for(int outw=0; outw < 2; ++outw) {
                int outpos = n * outgap[0] + k * outgap[1] + (y*2 + outh) * outgap[2] + (x*2 + outw); 
                // printf("\n outpos = %d", outpos);
                out[outpos] += Tile[outh * 2 + outw];
              }
          }
        }
      }
    }
  }
}