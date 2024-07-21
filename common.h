#pragma once
#include <arm_sve.h>
#define PRAGMA(X) _Pragma(#X)
#define OMP
#ifdef OMP
	#define PRAGMA_OMP_PARALLEL_FOR() PRAGMA(omp parallel for)
	#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(X) PRAGMA(omp parallel for collapse(X))
	#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE_PROC_BIND_SPREAD(X) PRAGMA(omp parallel for collapse(X) proc_bind(spread))
#else
	#define PRAGMA_OMP_PARALLEL_FOR() 
	#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(X) 
	#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE_PROC_BIND_SPREAD(X) 
#endif

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
#define FLT_H 3
#define FLT_W 3

#define TILE_IN_HW 6
#define TILE_IN_H 6
#define TILE_IN_W 6

#define TILE_OUT_HW 4
#define TILE_OUT_H 4
#define TILE_OUT_W 4


#define ROUND(A, B) ((A) / (B) * (B))
#define ROUND_UP(A, B) (((A) + (B) - 1) / (B) * (B))

#define DIVIDE(A, B) ((A) / (B))
#define DIVIDE_UP(A, B) (((A) + (B) - 1) / (B))

#define ALLOC_ALIGNMENT 4096  // Page size

#define OMP_GET_THREAD_ID() omp_get_thread_num()	// the thread id, not the number of threads
#define OMP_GET_MAX_THREADS() omp_get_max_threads()
#define PREFETCH
#ifdef PREFETCH
	#define PREFETCH_READ(X) __builtin_prefetch(&(X), 0)
	#define PREFETCH_WRITE(X) __builtin_prefetch(&(X), 1)
#else
	#define PREFETCH_READ(X)
	#define PREFETCH_WRITE(X)
#endif
#define ATTRIBUTE_ALIGN(X) __attribute__((aligned ((X))))

#define ALWAYS_INLINE inline __attribute__((always_inline))