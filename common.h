#pragma once
#include <arm_sve.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>

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

#define FLT_H 3
#define FLT_W 3
#define FLT_HW 3

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
#define DIV(A, B) ((A) / (B))
#define DIV_UP(A, B) (((A) + (B) - 1) / (B))

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


ALWAYS_INLINE double timestamp() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1.0e3 + tv.tv_usec * 1.e-3;
}
double times_ms[10];
double start_time;

// const int64_t k_blk_size = 64;
// const int64_t c_blk_size = 32;

typedef struct {
  int64_t b;
  int64_t th;
  int64_t tw;
} TileIndex;


// Tensors' shape, pass them when passing tensor as a pointer
typedef struct {
  int64_t oc;	// number of output channels
  int64_t ic;   // number of input channels
  int64_t h;
  int64_t w;
} FltShape;


typedef struct {
  int64_t numImg;
  int64_t ic;   // number of input channels
  int64_t h;
  int64_t w;
} ImgShape;

typedef struct {
  int64_t oc;
  int64_t ic;   // number of input channels
  int64_t h;
  int64_t w;
} UShape;

typedef struct {
  int64_t numTileTotal;
  int64_t ic;   // number of input channels
  int64_t h;
  int64_t w;
} VShape;

typedef struct {
  int64_t numImg;
  int64_t oc;   // number of output channels
  int64_t h;
  int64_t w;
} OutShape;

typedef struct {
  int64_t numImg;   // number of output channels
  int64_t numTilePerImg;
  int64_t numTileTotal;
  int64_t h;
  int64_t w;
} TileShape;

OutShape getOutShape(ImgShape is, FltShape fs) {
  OutShape os;
  os.numImg = is.numImg;
  os.oc = fs.oc;
  os.h = is.h - fs.h + 1;
  os.w = is.w - fs.w + 1;
  return os;
}

TileShape getTileShape(ImgShape is, OutShape os) {
  TileShape ts;
  ts.h = DIV_UP(os.h, TILE_OUT_H);
  ts.w = DIV_UP(os.w, TILE_OUT_W);
  ts.numImg = is.numImg;
  ts.numTilePerImg = ts.h * ts.w;
  ts.numTileTotal = ts.numTilePerImg * ts.numImg;
  return ts;
}

UShape getUShape(FltShape fs) {
  UShape us;
  us.oc = fs.oc;
  us.ic = fs.ic;
  us.h = TILE_IN_W;
  us.w = TILE_IN_W;
  return us;
}

VShape getVShape(ImgShape is, TileShape ts) {
  VShape vs;
  vs.numTileTotal = ts.numTileTotal;
  vs.ic = is.ic;
  vs.h = TILE_IN_H;
  vs.w = TILE_IN_W;
  return vs;
}

TileIndex getTileIndex(int64_t tileNo, TileShape ts) {
  TileIndex ti;
  ti.b = tileNo / ts.numTilePerImg;
  tileNo = tileNo % ts.numTilePerImg;
  ti.th = tileNo / ts.w;
  ti.tw = tileNo % ts.w;
  return ti;
}


typedef struct {   // Interval which is [start, end)
  int64_t start;
  int64_t end;
	int64_t len;
} Interval;


Interval newInterval(int64_t start, int64_t end) {
	Interval it;
	it.start = start;
	it.end = end;
	it.len = end - start;
  return it;
}

Interval newIntervalWithUpperBound(int64_t start, int64_t step, int64_t upperBound) {
	Interval it;
	it.start = start;
	it.end = MIN(start + step, upperBound);
  it.len = it.end - it.start;
  return it;
}

/* Parameters */

const int64_t outputChannelBlockSize = 64;
const int64_t inputChannelBlockSize = 64;
const int64_t tileBlockSize = 18;