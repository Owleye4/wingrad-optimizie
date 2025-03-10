CFLAG = -O3 -g -Wall -march=native -mavx2 -mfma -fopenmp

all:
	g++ driver.cc winograd.cc -std=c++11 ${CFLAG} -o winograd
