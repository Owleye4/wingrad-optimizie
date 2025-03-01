CFLAG = -O3 -g -Wall -fsanitize=address -fopenmp

all:
	g++ driver.cc winograd.cc -std=c++11 ${CFLAG} -o winograd
