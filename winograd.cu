#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <mkl.h>
#include <cublas_v2.h>
#include <Error.h>
#include "common.h"

ALWAYS_INLINE void filterOcIcPack(float* __restrict__ filter, FltShape fs, float* __restrict__ packedFilter) {
  int outChannel = fs.oc, inChannel = fs.ic;
  typedef float (*packedFilerTensor_t) [FLT_W][outChannel][inChannel];
  typedef float (*filterTensor_t) [inChannel][FLT_H][FLT_W];
  packedFilerTensor_t packedFilerTensor = (packedFilerTensor_t) packedFilter;
  filterTensor_t filterTensor = (filterTensor_t) filter;
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for(int k = 0; k < outChannel; ++k)
    for(int h = 0; h < FLT_HW; ++h)
      for(int w = 0; w < FLT_HW; ++w)
        for(int c = 0; c < inChannel; ++c) {
          packedFilerTensor[h][w][k][c] = filterTensor[k][c][h][w];
        }
}

ALWAYS_INLINE void ImageTileIcPack(float* __restrict__ image, ImgShape is,  float* __restrict__ packedImage,  TileShape ts) {
  int  inputChannels = is.ic, imgHeight = is.h, imgWidth = is.w, numTileTotal = ts.numTileTotal;
  typedef float (*ImgTensor_t) [inputChannels][imgHeight][imgWidth];
  typedef float (*packedImgTensor_t) [TILE_IN_W][numTileTotal][inputChannels];
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for(int tile = 0; tile < numTileTotal; ++tile)
    for(int ic = 0; ic < inputChannels; ++ic)
      for(int h = 0; h < TILE_IN_H; ++h)
        for(int w = 0; w < TILE_IN_W; ++w) {
          TileIndex ti = getTileIndex(tile, ts);
          int b = ti.b, x = ti.tw, y = ti.th;
          packedImgTensor_t packedImgTensor = (packedImgTensor_t) packedImage;
          ImgTensor_t imgTensor = (ImgTensor_t) image;
          if(y * 4 + h < imgHeight && x * 4 + w < imgWidth)
            packedImgTensor[h][w][tile][ic] = imgTensor[b][ic][y * 4 + h][x * 4 + w];
          else
            packedImgTensor[h][w][tile][ic] = 0;
        }
}

__global__ void srcTransformCUDA(float* __restrict__ packedImage, float* __restrict__ V, VShape vs, int simdDimSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float z0, z1, z2, z3, z4, z5, z6;
  while (idx < simdDimSize) {
    for (int w = 0; w < TILE_IN_W; ++w) {
      z6 = packedImage[0 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 = 4.0f * z6;

      z6 = packedImage[1 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z1 = -4.0f * z6;
      z2 =  4.0f * z6;
      z3 = -2.0f * z6;
      z4 =  2.0f * z6;
      z5 =  4.0f * z6;

      z6 = packedImage[2 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = packedImage[3 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z1 +=  z6;
      z2 += -z6;
      z3 +=  2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = packedImage[4 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = packedImage[5 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z5 += z6;

      V[0 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z0;
      V[1 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z1;
      V[2 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z2;
      V[3 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z3;
      V[4 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z4;
      V[5 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z5;
    }

    for (int h = 0; h < TILE_IN_H; ++h) {
      z6 = V[h * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx];

      z0 = 4.0f * z6;

      z6 = V[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx];

      z1 = -4.0f * z6;
      z2 =  4.0f * z6;
      z3 = -2.0f * z6;
      z4 =  2.0f * z6;
      z5 =  4.0f * z6;

      z6 = V[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = V[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx];

      z1 +=  z6;
      z2 += -z6;
      z3 +=  2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = V[h * TILE_IN_W * simdDimSize + 4 * simdDimSize + idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = V[h * TILE_IN_W * simdDimSize + 5 * simdDimSize + idx];

      z5 += z6;

      V[h * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx] = z0;
      V[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx] = z1;
      V[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx] = z2;
      V[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx] = z3;
      V[h * TILE_IN_W * simdDimSize + 4 * simdDimSize + idx] = z4;
      V[h * TILE_IN_W * simdDimSize + 5 * simdDimSize + idx] = z5;
    }
    idx += blockDim.x * gridDim.x;
  }
}

ALWAYS_INLINE void srcTransform(float* __restrict__ packedImage, float* __restrict__  V, VShape vs) {
  srcTransformCUDA<<<1, 256>>>(packedImage, V, vs, vs.ic * vs.numTileTotal);
  HANDLER_ERROR_MSG("kernel panic!!!");
}

__global__ void filterTransformCUDA(float* __restrict__ packedFilter, float* __restrict__ U, UShape us, int simdDimSize) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float z0, z1, z2, z3, z4, z5, z6;
  while (idx < simdDimSize) {
    for (int i = 0; i < FLT_HW; ++i) {
      z6 = packedFilter[0 * FLT_W * simdDimSize + i * simdDimSize + idx];

      z0 = (1.0f / 4.0f)  * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = packedFilter[1 * FLT_W * simdDimSize + i * simdDimSize + idx];

      z1 += (-1.0f / 6.0f)  * z6;
      z2 += ( 1.0f / 6.0f)  * z6;
      z3 += (1.0f  / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = packedFilter[2 * FLT_W * simdDimSize + i * simdDimSize + idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += ( 1.0f / 6.0f) * z6;
      z4 += ( 1.0f / 6.0f) * z6;
      z5 = z6;

      U[0 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z0;
      U[1 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z1;
      U[2 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z2;
      U[3 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z3;
      U[4 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z4;
      U[5 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z5;
    }

    for (int i = 0; i < TILE_IN_H; ++i) {
      z6 = U[i * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx];

      z0 = (1.0f / 4.0f)  * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = U[i * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx];

      z1 += (-1.0f / 6.0f)  * z6;
      z2 += ( 1.0f / 6.0f)  * z6;
      z3 += (1.0f  / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = U[i * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += ( 1.0f / 6.0f) * z6;
      z4 += ( 1.0f / 6.0f) * z6;
      z5 = z6;

      U[i * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx] = z0;
      U[i * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx] = z1;
      U[i * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx] = z2;
      U[i * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx] = z3;
      U[i * TILE_IN_W * simdDimSize + 4 * simdDimSize + idx] = z4;
      U[i * TILE_IN_W * simdDimSize + 5 * simdDimSize + idx] = z5;
    }
    idx += blockDim.x * gridDim.x;
  }
}

ALWAYS_INLINE void filterTransform(float* __restrict__ packedFilter, float* __restrict__ U, UShape us) {
  // PRAGMA_OMP_PARALLEL_FOR()
  // for(int ocic = 0; ocic < us.oc * us.ic; ocic += FP32_PER_REG) 
  filterTransformCUDA<<<1, 256>>>(packedFilter, U, us, us.ic * us.oc);
  HANDLER_ERROR_MSG("kernel panic!!!");
}

__global__ void destTransformCUDA(float* __restrict__ M, float* __restrict__ Y, int simdDimSize) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  float z0, z1, z2, z3, z4;
  while (idx < simdDimSize) {
    for (int w = 0; w < TILE_IN_W; ++w) {
      z4 = M[0 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 = z4;

      z4 = M[1 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 = z0 + z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;
      
      z4 = M[2 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = M[3 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = M[4 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = M[5 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z3 += z4;

      Y[0 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z0;
      Y[1 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z1;
      Y[2 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z2;
      Y[3 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z3;
    }

    for (int h = 0; h < TILE_OUT_HW; ++h) {
      z4 = Y[h * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx];

      z0 = z4;

      // z4 = svld1(pg, &YTensor[h][1][idx]);
      // z4 = YTensor[h][1][idx];
      z4 = Y[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx];

      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;
      
      // z4 = svld1(pg, &YTensor[h][2][idx]);
      // z4 = YTensor[h][2][idx];
      z4 = Y[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx];
      
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      // z4 = svld1(pg, &YTensor[h][3][idx]);
      // z4 = YTensor[h][3][idx];
      z4 = Y[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx];

      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      // z4 = svld1(pg, &YTensor[h][4][idx]);
      // z4 = YTensor[h][4][idx];
      z4 = Y[h * TILE_IN_W * simdDimSize + 4 * simdDimSize + idx];


      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      // z4 = svld1(pg, &YTensor[h][5][idx]);
      // z4 = YTensor[h][5][idx];
      z4 = Y[h * TILE_IN_W * simdDimSize + 5 * simdDimSize + idx];

      z3 += z4;

      // svst1_f32(pg, &YTensor[h][0][idx], z0);
      // svst1_f32(pg, &YTensor[h][1][idx], z1);
      // svst1_f32(pg, &YTensor[h][2][idx], z2);
      // svst1_f32(pg, &YTensor[h][3][idx], z3);
      // YTensor[h][0][idx] = z0;
      // YTensor[h][1][idx] = z1;
      // YTensor[h][2][idx] = z2;
      // YTensor[h][3][idx] = z3;
      Y[h * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx] = z0;
      Y[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx] = z1;
      Y[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx] = z2;
      Y[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx] = z3;
    }
    idx += blockDim.x * gridDim.x;
  }
}

ALWAYS_INLINE void destTransform(float* __restrict__ M, float* __restrict__ Y, int simdDimSize) {
  // for (int octile = 0; octile < simdDimSize; octile += FP32_PER_REG)
  destTransformCUDA<<<1, 256>>>(M, Y, simdDimSize);
  HANDLER_ERROR_MSG("kernel panic!!!");
}

ALWAYS_INLINE void destStore(float* __restrict__ Y, float* __restrict__ out, OutShape os,  TileShape ts) {
  typedef float (*YTensor_t) [TILE_IN_W][os.oc][ts.numTileTotal];
  typedef float (*outTensor_t) [os.oc][os.h][os.w];
  YTensor_t YTensor = (YTensor_t) Y;
  outTensor_t outTensor = (outTensor_t) out;
  for(int h = 0; h < TILE_OUT_H; ++h)
    for(int w = 0; w < TILE_OUT_W; ++w)
      for(int k = 0; k < os.oc; ++k)
        for(int b = 0; b < ts.numTileTotal; ++b) {
          TileIndex ti = getTileIndex(b, ts);
          int n = ti.b, x = ti.tw, y = ti.th;
          if(y * 4 + h < os.h && x * 4 + w < os.w) 
            outTensor[n][k][y * 4 + h][x * 4 + w] = YTensor[h][w][k][b];
        }
}

void winconv_2x3(float *__restrict__ image, const int inHeight,
                 const int inWidth, const int numInChannel, float *__restrict__ filter,
                 const int numOutChannel, const int numBatch, float *__restrict__ out,
                 float *__restrict__ U_no_use, float *__restrict__ V_no_use,
                 float *__restrict__ M_no_use) {

  /* new vars of shape */
  ImgShape  is = {numBatch, numInChannel, inHeight, inWidth};
  FltShape  fs = {numOutChannel, numInChannel, FLT_H, FLT_W};
  OutShape  os = getOutShape(is, fs);
  TileShape ts = getTileShape(is, os);
  UShape    us = getUShape(fs);
  VShape    vs = getVShape(is, ts);

  float* packedFilter =  (float*)  aligned_alloc(ALLOC_ALIGNMENT, sizeof(float) * FLT_H * FLT_W * us.oc * us.ic);
  assert(packedFilter != NULL);

  float* packedImage =  (float*)   aligned_alloc(ALLOC_ALIGNMENT, sizeof(float) * TILE_IN_H * TILE_IN_H * vs.numTileTotal * vs.ic);
  assert(packedImage  != NULL);

  float* packedFilterDevice, *packedImageDevice;
  HANDLER_ERROR_ERR(cudaMalloc(&packedFilterDevice, sizeof(float) * FLT_H * FLT_W * us.oc * us.ic));
  HANDLER_ERROR_ERR(cudaMalloc(&packedImageDevice , sizeof(float) * TILE_IN_H * TILE_IN_H * vs.numTileTotal * vs.ic)); 

  float * U, * V;
  HANDLER_ERROR_ERR(cudaMalloc(&U, sizeof(float) * TILE_IN_H * TILE_IN_W * us.oc * us.ic));
  HANDLER_ERROR_ERR(cudaMalloc(&V, sizeof(float) * TILE_IN_H * TILE_IN_W * vs.numTileTotal * vs.ic)); 

  typedef float (*UTensor_t) [TILE_IN_W][     us.oc     ][us.ic];
  typedef float (*VTensor_t) [TILE_IN_W][vs.numTileTotal][vs.ic];
  UTensor_t UTensor = (UTensor_t) U;
  VTensor_t VTensor = (VTensor_t) V;

  filterOcIcPack  (filter, fs, packedFilter   );
  ImageTileIcPack (image , is, packedImage, ts);


  HANDLER_ERROR_ERR(cudaMemcpy(packedFilterDevice, packedFilter, sizeof(float) * FLT_H * FLT_W * fs.oc * fs.ic, cudaMemcpyHostToDevice));
  HANDLER_ERROR_ERR(cudaMemcpy(packedImageDevice,  packedImage, sizeof(float) * TILE_IN_H * TILE_IN_H * vs.numTileTotal * vs.ic, cudaMemcpyHostToDevice));

  srcTransform(packedImageDevice, V, vs);
  filterTransform(packedFilterDevice, U, us);

  float *M, *Y, *YHost; 
  HANDLER_ERROR_ERR(cudaMalloc(&M, sizeof(float) * TILE_IN_H  * TILE_IN_W  * us.oc * vs.numTileTotal));
  HANDLER_ERROR_ERR(cudaMalloc(&Y, sizeof(float) * TILE_OUT_H * TILE_IN_W  * us.oc * vs.numTileTotal));
  YHost = (float*) aligned_alloc(ALLOC_ALIGNMENT, sizeof(float) * TILE_OUT_H * TILE_IN_W * vs.numTileTotal * us.oc);
  typedef float (*MTensor_t) [TILE_IN_W][us.oc][vs.numTileTotal];
  typedef float (*YTensor_t) [TILE_IN_W][us.oc][vs.numTileTotal];
  MTensor_t MTensor = (MTensor_t) M;

  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f, beta = 0.0f;
  for(int i = 0; i < TILE_IN_H * TILE_IN_W; ++i) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                vs.numTileTotal, us.oc, us.ic,
                &alpha,
                (float*)(VTensor[i/TILE_IN_W][i%TILE_IN_W]),
                vs.ic, 
                (float*)(UTensor[i/TILE_IN_W][i%TILE_IN_W]),
                us.ic, 
                &beta, 
                (float*)(MTensor[i/TILE_IN_W][i%TILE_IN_W]),
                vs.numTileTotal);
  }

  cublasDestroy(handle);

  destTransform((float *)M, (float *)Y, us.oc * vs.numTileTotal);

  cudaMemcpy(YHost, Y, sizeof(float) * TILE_OUT_H * TILE_IN_W * us.oc * vs.numTileTotal, cudaMemcpyDeviceToHost);

  destStore(YHost, out, os, ts);

  cudaFree(U);
  cudaFree(V);
  cudaFree(packedImageDevice);
  cudaFree(packedFilterDevice);
  cudaFree(M);
  cudaFree(Y);
  free(packedImage);
  free(packedFilter);
  free(YHost);
}