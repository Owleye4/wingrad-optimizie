KBLAS_NOLOCKING_LIB_DIR = /shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/lib/kblas/omp
KBLAS_NOLOCKING_INCLUDE_DIR = /shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/include/
all:
	clang -std=c11 -fopenmp -O3 -g driver.c winograd.c -o winograd -flto -march=native \
			-I ${KBLAS_NOLOCKING_INCLUDE_DIR} -L${KBLAS_NOLOCKING_LIB_DIR} -lkblas
	# gcc -std=c11 -D__DEBUG -O0 -g driver.c winograd.c -o winograd