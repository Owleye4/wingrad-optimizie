CFLAG_OPT = -O3 -march=native -ffast-math -fvectorize -funroll-loops -fopenmp -mtune=native

all:
	icx -std=c11 -O3 -g driver.c winograd.c -o winograd ${CFLAG_OPT} -lmkl_rt

debug:
	clang -std=c11 -Og -g driver.c winograd.c -o winograd -march=native+sve # for perf & debug