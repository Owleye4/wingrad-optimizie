CFLAG_OPT = -O3

all:
	nvcc -std=c++11 -O3 -g driver.cc winograd.cu -o winograd ${CFLAG_OPT} -lcublas
