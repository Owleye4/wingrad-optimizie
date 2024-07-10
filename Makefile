all:
	# Only Bisheng compiler (which uses clang) is allowed according to the PAC commitee.
	clang -std=c11 -Ofast -g driver.c winograd.c -o winograd -flto -march=native  -mtune=native -static -fopenmp -ffast-math
