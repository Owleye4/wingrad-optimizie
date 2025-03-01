#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

void image_transform(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     V_shape_t vs,
                     int collapsed_dim_size) {
  typedef float(*packed_image_tensor_t)[TILE_IN_W][collapsed_dim_size];
  typedef float(*V_tensor_t)[TILE_IN_W][collapsed_dim_size];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  float z0, z1, z2, z3, z4, z5, z6;
#pragma omp parallel for private(z0, z1, z2, z3, z4, z5, z6)
  for (int idx = 0; idx < collapsed_dim_size; idx++) {
    for (int w = 0; w < TILE_IN_W; ++w) {
      z6 = packed_image_tensor[0][w][idx];

      z0 = 4.0f * z6;

      z6 = packed_image_tensor[1][w][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = packed_image_tensor[2][w][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = packed_image_tensor[3][w][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = packed_image_tensor[4][w][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = packed_image_tensor[5][w][idx];

      z5 += z6;

      V_tensor[0][w][idx] = z0;
      V_tensor[1][w][idx] = z1;
      V_tensor[2][w][idx] = z2;
      V_tensor[3][w][idx] = z3;
      V_tensor[4][w][idx] = z4;
      V_tensor[5][w][idx] = z5;
    }

    for (int h = 0; h < TILE_IN_H; ++h) {
      z6 = V_tensor[h][0][idx];

      z0 = 4.0f * z6;

      z6 = V_tensor[h][1][idx];

      z1 = -4.0f * z6;
      z2 = 4.0f * z6;
      z3 = -2.0f * z6;
      z4 = 2.0f * z6;
      z5 = 4.0f * z6;

      z6 = V_tensor[h][2][idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = V_tensor[h][3][idx];

      z1 += z6;
      z2 += -z6;
      z3 += 2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = V_tensor[h][4][idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = V_tensor[h][5][idx];

      z5 += z6;

      V_tensor[h][0][idx] = z0;
      V_tensor[h][1][idx] = z1;
      V_tensor[h][2][idx] = z2;
      V_tensor[h][3][idx] = z3;
      V_tensor[h][4][idx] = z4;
      V_tensor[h][5][idx] = z5;
    }
  }
}

void filter_transform(float *__restrict__ packed_filter,
                      filter_shape_t fs,
                      float *__restrict__ U,
                      U_shape_t us,
                      int collapsed_dim_size) {
  typedef float(*packed_filter_tensor_t)[fs.w][collapsed_dim_size];
  typedef float(*U_tensor_t)[us.w][collapsed_dim_size];
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
  U_tensor_t U_tensor = (U_tensor_t)U;

  float z0, z1, z2, z3, z4, z5, z6;
#pragma omp parallel for private(z0, z1, z2, z3, z4, z5, z6)
  for (int idx = 0; idx < collapsed_dim_size; idx++) {
    for (int w = 0; w < fs.w; ++w) {
      z6 = packed_filter_tensor[0][w][idx];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = packed_filter_tensor[1][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = packed_filter_tensor[2][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[0][w][idx] = z0;
      U_tensor[1][w][idx] = z1;
      U_tensor[2][w][idx] = z2;
      U_tensor[3][w][idx] = z3;
      U_tensor[4][w][idx] = z4;
      U_tensor[5][w][idx] = z5;
    }

    for (int h = 0; h < us.h; ++h) {
      z6 = U_tensor[h][0][idx];

      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = U_tensor[h][1][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = U_tensor[h][2][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[h][0][idx] = z0;
      U_tensor[h][1][idx] = z1;
      U_tensor[h][2][idx] = z2;
      U_tensor[h][3][idx] = z3;
      U_tensor[h][4][idx] = z4;
      U_tensor[h][5][idx] = z5;
    }
  }
}

void dest_transform(float *__restrict__ M, float *__restrict__ Y, tiling_info_t ti, int collapsed_dim_size) {
  typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  M_tensor_t M_tensor = (M_tensor_t)M;
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  float z0, z1, z2, z3, z4;
#pragma omp parallel for private(z0, z1, z2, z3, z4)
  for (int idx = 0; idx < collapsed_dim_size; idx++) {
    for (int w = 0; w < ti.tile_in_w; ++w) {
      z4 = M_tensor[0][w][idx];
      z0 = z4;

      z4 = M_tensor[1][w][idx];
      z0 = z0 + z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = M_tensor[2][w][idx];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = M_tensor[3][w][idx];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = M_tensor[4][w][idx];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = M_tensor[5][w][idx];
      z3 += z4;

      Y_tensor[0][w][idx] = z0;
      Y_tensor[1][w][idx] = z1;
      Y_tensor[2][w][idx] = z2;
      Y_tensor[3][w][idx] = z3;
    }

    for (int h = 0; h < ti.tile_out_h; ++h) {
      z4 = Y_tensor[h][0][idx];

      z0 = z4;

      z4 = Y_tensor[h][1][idx];
      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = Y_tensor[h][2][idx];
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = Y_tensor[h][3][idx];
      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = Y_tensor[h][4][idx];
      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = Y_tensor[h][5][idx];

      z3 += z4;

      Y_tensor[h][0][idx] = z0;
      Y_tensor[h][1][idx] = z1;
      Y_tensor[h][2][idx] = z2;
      Y_tensor[h][3][idx] = z3;
    }
  }
}

void filter_packing(float *__restrict__ filter, filter_shape_t fs, float *__restrict__ packedFilter) {
  typedef float(*filter_tensor_t)[fs.ic][fs.h][fs.w];
  typedef float(*packed_filter_tensor_t)[fs.w][fs.oc][fs.ic];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packedFilter;

#pragma omp parallel for collapse(3)
  for (int h = 0; h < fs.h; ++h)
    for (int w = 0; w < fs.w; ++w)
      for (int oc = 0; oc < fs.oc; oc++)
        for (int ic = 0; ic < fs.ic; ic++) 
          packed_filter_tensor[h][w][oc][ic] = filter_tensor[oc][ic][h][w];
}

void image_packing(float *__restrict__ image,
                   image_shape_t is,
                   float *__restrict__ packed_image,
                   tiling_info_t ts) {
  typedef float(*packedImage_tensor_t)[TILE_IN_W][ts.num_tiles_total][is.ic];
  typedef float(*image_tensor_t)[is.ic][is.h][is.w];
  packedImage_tensor_t packed_image_tensor = (packedImage_tensor_t)packed_image;
  image_tensor_t image_tensor = (image_tensor_t)image;

#pragma omp parallel for collapse(3)
  for (int tile = 0; tile < ts.num_tiles_total; tile++) {
    for (int ic = 0; ic < is.ic; ic++) {
      for (int h = 0; h < TILE_IN_H; ++h) {
        for (int w = 0; w < TILE_IN_W; ++w) {
          tile_index_t ti = get_tile_index(tile, ts);
          int batch = ti.b, ww = ti.tw, hh = ti.th;
          if (hh * 4 + h < is.h && ww * 4 + w < is.w)
            packed_image_tensor[h][w][tile][ic] = image_tensor[batch][ic][(hh * 4 + h)][(ww * 4 + w)];
          else
            packed_image_tensor[h][w][tile][ic] = 0;
        }
      }
    }
  }
}

void dest_unpacking_store(float *__restrict__ Y, float *__restrict__ out, out_shape_t os, tiling_info_t ti) {
  typedef float(*Y_tensor_t)[TILE_IN_W][os.oc][ti.num_tiles_total];
  typedef float(*out_tensor_t)[os.oc][os.h][os.w];
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  out_tensor_t out_tensor = (out_tensor_t)out;

  for (int h = 0; h < ti.tile_out_h; ++h) {
    for (int w = 0; w < ti.tile_out_w; ++w) {
      for (int oc = 0; oc < os.oc; oc++) {
        for (int tile = 0; tile < ti.num_tiles_total; tile++) {
          tile_index_t tidx = get_tile_index(tile, ti);
          int batch = tidx.b, ww = tidx.tw, hh = tidx.th;
          if (hh * 4 + h < os.h && ww * 4 + w < os.w)
            out_tensor[batch][oc][(hh * 4 + h)][(ww * 4 + w)] = Y_tensor[h][w][oc][tile];
        }
      }
    }
  }
}

void sgemm(int M, int N, int K, float *A, float *B, float *C) {
  typedef float(*A_tensor_t)[K];
  typedef float(*B_tensor_t)[K];
  typedef float(*C_tensor_t)[M];
  A_tensor_t A_tensor = (A_tensor_t)A;
  B_tensor_t B_tensor = (B_tensor_t)B;
  C_tensor_t C_tensor = (C_tensor_t)C;

#pragma omp parallel for collapse(2)
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      C_tensor[n][m] = 0;
      for (int k = 0; k < K; ++k) {
        C_tensor[n][m] += A_tensor[m][k] * B_tensor[n][k];
      }
    }
  }
}

void winconv(float *__restrict__ image,
             const int image_height,
             const int image_width,
             const int input_channel_num,
             float *__restrict__ filter,
             const int output_channel_num,
             const int batch_num,
             float *__restrict__ out,
             float *__restrict__ U_no_use,
             float *__restrict__ V_no_use,
             float *__restrict__ M_no_use) {
  /* new vars of shape */
  image_shape_t is = {batch_num, input_channel_num, image_height, image_width};
  filter_shape_t fs = {output_channel_num, input_channel_num, FLT_H, FLT_W};
  out_shape_t os = get_output_shape(is, fs);
  tiling_info_t ti = get_tiling_info(is, os);
  U_shape_t us = get_U_shape(fs);
  V_shape_t vs = get_V_shape(is, ti);

  float *packed_filter = (float *)malloc(sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
  float *packed_image = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles_total * is.ic);
  float *U = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
  float *V = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
  float *M = (float *)malloc(sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
  float *Y = (float *)malloc(sizeof(float) * os.h * os.w * os.oc * ti.num_tiles_total);

  filter_packing(filter, fs, packed_filter);
  filter_transform(packed_filter, fs, U, us, us.oc * us.ic);

  image_packing(image, is, packed_image, ti);
  image_transform(packed_image, V, vs, vs.ic * vs.num_tiles);

  for (int h = 0; h < ti.tile_in_h; ++h) {
    for (int w = 0; w < ti.tile_in_w; ++w) {
      typedef float(*U_tensor_t)[ti.tile_in_w][us.oc][us.ic];
      typedef float(*V_tensor_t)[ti.tile_in_w][vs.num_tiles][vs.ic];
      typedef float(*M_tensor_t)[ti.tile_in_w][us.oc][vs.num_tiles];
      U_tensor_t U_tensor = (U_tensor_t)U;
      V_tensor_t V_tensor = (V_tensor_t)V;
      M_tensor_t M_tensor = (M_tensor_t)M;
      sgemm(vs.num_tiles,
            us.oc,
            us.ic,
            (float *)(V_tensor[h][w]),
            (float *)(U_tensor[h][w]),
            (float *)(M_tensor[h][w]));
    }
  }

  dest_transform(M, Y, ti, us.oc * vs.num_tiles);
  dest_unpacking_store(Y, out, os, ti);

  free(packed_filter);
  free(packed_image);
  free(U);
  free(V);
  free(M);
  free(Y);
}