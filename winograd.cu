#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <arm_sve.h>
#include "common.h"
// #include "kblas.h"
#include <mkl.h>
ALWAYS_INLINE void filterIcPack(float* __restrict__ filter, FltShape fs, float* __restrict__ packedFiler) {
  typedef float (*packedFilerTensor_t) [FLT_H][FLT_W][fs.ic];
  typedef float (*filterTensor_t) [fs.ic][FLT_H][FLT_W];
  packedFilerTensor_t packedFilerTensor = (packedFilerTensor_t) packedFiler;
  filterTensor_t filterTensor = (filterTensor_t) filter;
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
  for(int  k = 0; k < fs.oc; ++k)
    for(int  h = 0; h < FLT_HW; ++h)
      for(int  w = 0; w < FLT_HW; ++w)
        for(int  c = 0; c < fs.ic; ++c)
          packedFilerTensor[k][h][w][c] = filterTensor[k][c][h][w];
}

ALWAYS_INLINE void ImageIcPack(float* __restrict__ image, ImgShape is,  float* __restrict__ packedImage) {
  typedef float (*packedImageTensor_t) [is.h][is.w][is.ic];
  typedef float (*imageTensor_t) [is.ic][is.h][is.w];
  packedImageTensor_t packedImageTensor = (packedImageTensor_t) packedImage;
  imageTensor_t imageTensor = (imageTensor_t) image;
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for(int  n = 0; n < is.numImg; ++n)
    for(int  h = 0; h < is.h; ++h)
      for(int  c = 0; c < is.ic; ++c)
        for(int  w = 0; w < is.w; ++w)
          packedImageTensor[n][h][w][c] = imageTensor[n][c][h][w];
}

ALWAYS_INLINE void filterTransformSVE(float* __restrict__ packedFilter, float* __restrict__ U, UShape us, int  simdDimSize, int  simdDimIndex) {
  typedef float (*packedFilterTensor_t) [FLT_W][simdDimSize];
  typedef float (*UTensor_t) [TILE_IN_W][simdDimSize];
  packedFilterTensor_t packedFilterTensor = (packedFilterTensor_t) packedFilter;
  UTensor_t UTensor = (UTensor_t) U;
  float tmp[TILE_IN_W][FLT_W][FP32_PER_REG] ATTRIBUTE_ALIGN(128);
  DECLARE_SVE_FP32_REGS();
  z25 = svdup_f32(1.0f);
  z26 = svdup_f32( -1.0f / 6.0f  );
  z27 = svdup_f32( -1.0f / 12.0f );
  z28 = svdup_f32(  1.0f / 4.0f  );
  z29 = svdup_f32(  1.0f / 6.0f  );
  z30 = svdup_f32(  1.0f / 12.0f );
  z31 = svdup_f32(  1.0f / 24.0f  );
  svbool_t pg = svwhilelt_b32(0, MIN(FP32_PER_REG, simdDimSize-simdDimIndex));
  // G * filter
  for (int  i = 0; i < FLT_HW; ++i) {     // 这个循环按row遍历filter（按G以及结果的column）， 按列产生结果。
    z6 = svld1(pg, &packedFilterTensor[0][i][simdDimIndex]);

    z0 = svmul_f32_x(pg, z28, z6);
    z1 = svmul_f32_x(pg, z26, z6);
    z2 = svmul_f32_x(pg, z26, z6);
    z3 = svmul_f32_x(pg, z31, z6);
    z4 = svmul_f32_x(pg, z31, z6);

    z6 = svld1(pg, &packedFilterTensor[1][i][simdDimIndex]);

    // z0 += 0;
    z1 = svmla_f32_x(pg, z1, z26, z6);
    z2 = svmla_f32_x(pg, z2, z29, z6);
    z3 = svmla_f32_x(pg, z3, z30, z6);
    z4 = svmla_f32_x(pg, z4, z27, z6);
    // z5 += 0;
    z6 = svld1(pg, &packedFilterTensor[2][i][simdDimIndex]);

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
  for (int  i = 0; i < TILE_IN_H; ++i) {    // 这个循环按row遍历(G * filter)（按G_T的column遍历），按行产生结果。
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

    svst1_f32(pg, &UTensor[i][0][simdDimIndex], z0);
    svst1_f32(pg, &UTensor[i][1][simdDimIndex], z1);
    svst1_f32(pg, &UTensor[i][2][simdDimIndex], z2);
    svst1_f32(pg, &UTensor[i][3][simdDimIndex], z3);
    svst1_f32(pg, &UTensor[i][4][simdDimIndex], z4);
    svst1_f32(pg, &UTensor[i][5][simdDimIndex], z5);
  }
}


ALWAYS_INLINE void srcPaddingAndTransformSVE(float* __restrict__ packedImage, ImgShape is,  float* V, VShape vs, int  tileNo, TileShape ts, int  c) {
  TileIndex ti = getTileIndex(tileNo, ts);
  int64_t  n = ti.b, x = ti.tw, y = ti.th;
  int64_t  inHeight = is.h, inWidth = is.w, C = is.ic;
  typedef float (*packedImageTensor_t) [is.h][is.w][is.ic];
  typedef float (*VSlicedTensor_t) [TILE_IN_W][is.ic];
  packedImageTensor_t packedImageTensor = (packedImageTensor_t) packedImage;
  VSlicedTensor_t VTensor = (VSlicedTensor_t) (V + tileNo * TILE_IN_H * TILE_IN_W * is.ic);
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
  memset((void*) tmp, 0, sizeof(tmp));
  svbool_t pg = svwhilelt_b32(0L, MIN(FP32_PER_REG, is.ic - c));

  for (int  xx = 0; xx < TILE_IN_W && (x * 4 + xx) < is.w; ++xx) {   // 按列产生结果。
    if((y * 4 + 0) < inHeight){
      z6 = svld1(pg, &packedImageTensor[n][y * 4 + 0][x * 4 + xx][c]);
      z0 = svmul_f32_x(pg, z28, z6);
    }

    if((y * 4 + 1) < is.h) {
      z6 = svld1(pg, &packedImageTensor[n][y * 4 + 1][x * 4 + xx][c]);
      z1 = svmul_f32_x(pg, z29, z6);
      z2 = svmul_f32_x(pg, z28, z6);
      z3 = svmul_f32_x(pg, z27, z6);
      z4 = svmul_f32_x(pg, z26, z6);
      z5 = svmul_f32_x(pg, z28, z6);
    }

    if((y * 4 + 2) < is.h) {
      z6 = svld1(pg, &packedImageTensor[n][y * 4 + 2][x * 4 + xx][c]);
      z0 = svmla_f32_x(pg, z0, z31, z6);
      z1 = svmla_f32_x(pg, z1, z29, z6);
      z2 = svmla_f32_x(pg, z2, z29, z6);
      z3 = svmla_f32_x(pg, z3, z25, z6);
      z4 = svmla_f32_x(pg, z4, z25, z6);
    }

    if((y * 4 + 3) < is.h) {
      z6 = svld1(pg, &packedImageTensor[n][y * 4 + 3][x * 4 + xx][c]);
      z1 = svmla_f32_x(pg, z1, z24, z6);
      z2 = svmla_f32_x(pg, z2, z25, z6);
      z3 = svmla_f32_x(pg, z3, z26, z6);
      z4 = svmla_f32_x(pg, z4, z27, z6);
      z5 = svmla_f32_x(pg, z5, z31, z6);
    }

    if((y * 4 + 4) < is.h) {
      z6 = svld1(pg, &packedImageTensor[n][y * 4 + 4][x * 4 + xx][c]);
      z0 = svmla_f32_x(pg, z0, z24, z6);
      z1 = svmla_f32_x(pg, z1, z24, z6);
      z2 = svmla_f32_x(pg, z2, z24, z6);
      z3 = svmla_f32_x(pg, z3, z24, z6);
      z4 = svmla_f32_x(pg, z4, z24, z6);
    }

    if((y * 4 + 5) < is.h) {
      z6 = svld1(pg, &packedImageTensor[n][y * 4 + 5][x * 4 + xx][c]);
      z5 = svmla_f32_x(pg, z5, z24, z6);
    }

    svst1_f32(pg, tmp[0][xx], z0);
    svst1_f32(pg, tmp[1][xx], z1);
    svst1_f32(pg, tmp[2][xx], z2);
    svst1_f32(pg, tmp[3][xx], z3);
    svst1_f32(pg, tmp[4][xx], z4);
    svst1_f32(pg, tmp[5][xx], z5);
  }

  for (int  yy = 0; yy < TILE_IN_H; ++yy) {   // 按行产生结果。
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

    svst1_f32(pg, &VTensor[yy][0][c], z0);
    svst1_f32(pg, &VTensor[yy][1][c], z1);
    svst1_f32(pg, &VTensor[yy][2][c], z2);
    svst1_f32(pg, &VTensor[yy][3][c], z3);
    svst1_f32(pg, &VTensor[yy][4][c], z4);
    svst1_f32(pg, &VTensor[yy][5][c], z5);
  }
}

ALWAYS_INLINE void filterTransform(float* __restrict__ packedFilter, float* __restrict__ U, UShape us) {
  int  K = us.oc, C = us.ic;
  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for (int  k = 0; k < K; ++k) {
    for(int  c = 0; c < C; c += FP32_PER_REG) {
      filterTransformSVE(packedFilter + k * FLT_H * FLT_W * C, U + k * TILE_IN_H * TILE_IN_W * C, us, C, c);
    }
  }
}

ALWAYS_INLINE void destTransformSVE(float* __restrict__ M, float* __restrict__ Y, int64_t  simdDimSize, int64_t  simdDimIndex) {
  
  typedef float (*MTensor_t)[TILE_IN_W][simdDimSize];
  typedef float (*YTensor_t)[TILE_IN_W][simdDimSize];

  MTensor_t MTensor = (MTensor_t) M;
  YTensor_t YTensor = (YTensor_t) Y;

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
  svbool_t pg = svwhilelt_b32(0L, MIN(FP32_PER_REG, simdDimSize - simdDimIndex));
  for (int  w = 0; w < TILE_IN_W; ++w) {   // 按列产生结果。
    z4 = svld1(pg, &MTensor[0][w][simdDimIndex]);
    
    z0 = z4;

    z4 = svld1(pg, &MTensor[1][w][simdDimIndex]);
    
    z0 = svadd_f32_x(pg, z0, z4);
    z1 = z4;
    z2 = z4;
    z3 = z4;
    
    z4 = svld1(pg, &MTensor[2][w][simdDimIndex]);
    
    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svsub_f32_x(pg, z1, z4);
    z2 = svadd_f32_x(pg, z2, z4);
    z3 = svsub_f32_x(pg, z3, z4);

    z4 = svld1(pg, &MTensor[3][w][simdDimIndex]);

    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svmla_f32_x(pg, z1, z26, z4);
    z2 = svmla_f32_x(pg, z2, z28, z4);
    z3 = svmla_f32_x(pg, z3, z23, z4);

    z4 = svld1(pg, &MTensor[4][w][simdDimIndex]);

    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svmla_f32_x(pg, z1, z27, z4);
    z2 = svmla_f32_x(pg, z2, z28, z4);
    z3 = svmla_f32_x(pg, z3, z22, z4);

    z4 = svld1(pg, &MTensor[5][w][simdDimIndex]);

    z3 = svadd_f32_x(pg, z3, z4);

    svst1_f32(pg, &YTensor[0][w][simdDimIndex], z0);
    svst1_f32(pg, &YTensor[1][w][simdDimIndex], z1);
    svst1_f32(pg, &YTensor[2][w][simdDimIndex], z2);
    svst1_f32(pg, &YTensor[3][w][simdDimIndex], z3);
  }

  for (int  h = 0; h < TILE_OUT_HW; ++h) {   // 按行产生结果。
    z4 = svld1(pg, &YTensor[h][0][simdDimIndex]);
    
    z0 = z4;

    z4 = svld1(pg, &YTensor[h][1][simdDimIndex]);
    
    z0 = svadd_f32_x(pg, z0, z4);
    z1 = z4;
    z2 = z4;
    z3 = z4;
    
    z4 = svld1(pg, &YTensor[h][2][simdDimIndex]);
    
    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svsub_f32_x(pg, z1, z4);
    z2 = svadd_f32_x(pg, z2, z4);
    z3 = svsub_f32_x(pg, z3, z4);

    z4 = svld1(pg, &YTensor[h][3][simdDimIndex]);

    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svmla_f32_x(pg, z1, z26, z4);
    z2 = svmla_f32_x(pg, z2, z28, z4);
    z3 = svmla_f32_x(pg, z3, z23, z4);

    z4 = svld1(pg, &YTensor[h][4][simdDimIndex]);

    z0 = svadd_f32_x(pg, z0, z4);
    z1 = svmla_f32_x(pg, z1, z27, z4);
    z2 = svmla_f32_x(pg, z2, z28, z4);
    z3 = svmla_f32_x(pg, z3, z22, z4);  // BUG Here 

    z4 = svld1(pg, &YTensor[h][5][simdDimIndex]);

    z3 = svadd_f32_x(pg, z3, z4);

    svst1_f32(pg, &YTensor[h][0][simdDimIndex], z0);
    svst1_f32(pg, &YTensor[h][1][simdDimIndex], z1);
    svst1_f32(pg, &YTensor[h][2][simdDimIndex], z2);
    svst1_f32(pg, &YTensor[h][3][simdDimIndex], z3);
  }
}

ALWAYS_INLINE void destTransform(float* __restrict__ M, float* __restrict__ Y, int  simdDimSize) {
  for (int  simdDimIndex = 0; simdDimIndex < simdDimSize; simdDimIndex += FP32_PER_REG)
    destTransformSVE(M, Y, simdDimSize, simdDimIndex);
}

ALWAYS_INLINE void destStore(float* __restrict__ Y, float* __restrict__ out, OutShape os, Interval RangeOC, Interval RangeTile, TileShape ts) {
  typedef float (*YTensor_t) [TILE_IN_W][RangeOC.len][RangeTile.len];
  typedef float (*outTensor_t) [os.oc][os.h][os.w];
  YTensor_t YTensor = (YTensor_t) Y;
  outTensor_t outTensor = (outTensor_t) out;
  for(int  h = 0; h < TILE_OUT_H; ++h)
    for(int  w = 0; w < TILE_OUT_W; ++w)
      for(int  k = 0; k < RangeOC.len; ++k)
        for(int  b = 0; b < RangeTile.len; ++b) {
          TileIndex ti = getTileIndex(RangeTile.start + b, ts);
          int  n = ti.b, x = ti.tw, y = ti.th;
          if(y * 4 + h < os.h && x * 4 + w < os.w) 
            outTensor[n][RangeOC.start + k][y * 4 + h][x * 4 + w] = YTensor[h][w][k][b];
        }
}

void winconv_2x3(float *__restrict__ image, const int  inHeight,
                 const int  inWidth, const int  numInChannel, float *__restrict__ filter,
                 const int  numOutChannel, const int  numBatch, float *__restrict__ out,
                 float *__restrict__ U_no_use, float *__restrict__ V_no_use,
                 float *__restrict__ M_no_use) {

  /* new vars of shape */
  ImgShape is = {numBatch, numInChannel, inHeight, inWidth};
  FltShape fs = {numOutChannel, numInChannel, FLT_H, FLT_W};
  OutShape os = getOutShape(is, fs);
  TileShape ts = getTileShape(is, os);
  UShape us = getUShape(fs);
  VShape vs = getVShape(is, ts);
  float* filerPacked =  (float*)  aligned_alloc(ALLOC_ALIGNMENT, numOutChannel * numInChannel * FLT_H * FLT_W * sizeof(float));
  assert(filerPacked != NULL);
  float* imagePacked =  (float*)  aligned_alloc(ALLOC_ALIGNMENT, sizeof(float) * numBatch * numInChannel * inHeight * inWidth);
  assert(imagePacked != NULL);
  float* V = (float*)  aligned_alloc(ALLOC_ALIGNMENT, sizeof(float) * vs.numTileTotal  * TILE_IN_H * TILE_IN_W * numInChannel);
  assert(V != NULL);
  float* U = (float*)  aligned_alloc(ALLOC_ALIGNMENT, sizeof(float) * numOutChannel * TILE_IN_H * TILE_IN_W * numInChannel);
  assert(U != NULL);

  typedef float (*UTensor_t) [TILE_IN_H][TILE_IN_W][numInChannel];
  typedef float (*VTensor_t) [TILE_IN_H][TILE_IN_W][numInChannel];
  UTensor_t UTensor = (UTensor_t) U;
  VTensor_t VTensor = (VTensor_t) V;

  filterIcPack(filter, fs, filerPacked);
  ImageIcPack(image, is, imagePacked);

  filterTransform(filerPacked, U, us);

  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for (int  tileNo = 0; tileNo < vs.numTileTotal; ++tileNo) {
    for (int  c = 0; c < numInChannel; c += FP32_PER_REG) {
      srcPaddingAndTransformSVE(imagePacked, is, V, vs, tileNo, ts, c);
    }
  }

  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
  for(int  ocBlockStart = 0; ocBlockStart < numOutChannel; ocBlockStart +=  outputChannelBlockSize) {
    for(int  tileBlockStart = 0; tileBlockStart < ts.numTileTotal; tileBlockStart += tileBlockSize) {
      Interval RangeOC = newIntervalWithUpperBound(ocBlockStart, outputChannelBlockSize, numOutChannel);
      Interval RangeTile = newIntervalWithUpperBound(tileBlockStart, tileBlockSize, ts.numTileTotal);
      float M[TILE_IN_H  *  TILE_IN_W][RangeOC.len][RangeTile.len] ATTRIBUTE_ALIGN(128); memset(M, 0, sizeof(M));
      float Y[TILE_OUT_H *  TILE_IN_W][RangeOC.len][RangeTile.len] ATTRIBUTE_ALIGN(128);
      for(int  icBlockStart = 0; icBlockStart < numInChannel; icBlockStart += inputChannelBlockSize) {
        Interval RangeIC = newIntervalWithUpperBound(icBlockStart, inputChannelBlockSize, numInChannel);
        for(int  i = 0; i < TILE_IN_H * TILE_IN_W; ++i) {
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                      RangeOC.len, RangeTile.len, RangeIC.len, 1.0f, 
                      &UTensor[RangeOC.start][i/TILE_IN_W][i%TILE_IN_W][RangeIC.start], TILE_IN_H * TILE_IN_W * numInChannel, 
                      &VTensor[RangeTile.start][i/TILE_IN_W][i%TILE_IN_W][RangeIC.start], TILE_IN_H * TILE_IN_W * numInChannel, 
                      1.0f, (float*)M[i], RangeTile.len);
        }
      }
      destTransform((float *)M, (float *)Y, RangeOC.len * RangeTile.len);
      destStore((float *)Y, out, os, RangeOC, RangeTile, ts);
    }
  }

  free(U);
  free(V);
  free(imagePacked);
  free(filerPacked);
}