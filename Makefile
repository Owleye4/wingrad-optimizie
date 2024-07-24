KBLAS_LIB_DIR = /shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/lib/kblas/omp
KBLAS_INCLUDE_DIR = /shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/include

all:
	# Only Bisheng compiler (which uses clang) is allowed according to the PAC commitee.
	clang -std=c11 -Ofast -g driver.c winograd.c -o winograd  -mcpu=linxicore9100 -mtune=native  -ffast-math -fvectorize -funroll-loops -fopenmp \
		            -I ${KBLAS_INCLUDE_DIR} -L${KBLAS_LIB_DIR} -lkblas

debug:
	clang -std=c11 -Og -g driver.c winograd.c -o winograd -march=native+sve # for perf & debug
