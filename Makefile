all:
	# Only Bisheng compiler (which uses clang) is allowed according to the PAC commitee.
	clang -std=c11 -Ofast -g driver.c winograd.c -o winograd  -mcpu=linxicore9100 -mtune=native -static -ffast-math -fvectorize -funroll-loops -fopenmp
	# clang -std=c11 -Og -g driver.c winograd.c -o winograd -march=native+sve # for perf
