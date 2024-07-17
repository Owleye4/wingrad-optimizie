#pragma once
#include <arm_sve.h>
#define PRAGMA(X) _Pragma(#X)
#define PRAGMA_OMP_PARALLEL_FOR() PRAGMA(omp parallel for)
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(X) PRAGMA(omp parallel for collapse(X))

#define FP32_PER_REG 8

#define DECLARE_SVE_FP32_REGS() \
		svfloat32_t	z0; \
		svfloat32_t	z1; \
		svfloat32_t	z2; \
		svfloat32_t	z3; \
		svfloat32_t	z4; \
		svfloat32_t	z5; \
		svfloat32_t	z6; \
		svfloat32_t	z7; \
		svfloat32_t	z8; \
		svfloat32_t	z9; \
		svfloat32_t	z10; \
		svfloat32_t	z11; \
		svfloat32_t	z12; \
		svfloat32_t	z13; \
		svfloat32_t	z14; \
		svfloat32_t	z15; \
		svfloat32_t	z16; \
		svfloat32_t	z17; \
		svfloat32_t	z18; \
		svfloat32_t	z19; \
		svfloat32_t	z20; \
		svfloat32_t	z21; \
		svfloat32_t	z22; \
		svfloat32_t	z23; \
		svfloat32_t	z24; \
		svfloat32_t	z25; \
		svfloat32_t	z26; \
		svfloat32_t	z27; \
		svfloat32_t	z28; \
		svfloat32_t	z29; \
		svfloat32_t	z30; \
		svfloat32_t	z31;

#define MIN(A, B) (((A) < (B)) ? (A) : (B))
#define MAX(A, B) (((A) > (B)) ? (A) : (B))

#define FLT_HW 3
#define TILE_OUT_HW 4
#define TILE_IN_HW 6

