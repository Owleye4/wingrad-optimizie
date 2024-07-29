#!/bin/bash
#SBATCH -p pac
#SBATCH -n 1
#SBATCH -o output/job%j.out
#SBATCH -e error/job%j.err
#SBATCH -c 160
#SBATCH --exclusive
export OMP_NUM_THREADS=160
export LD_LIBRARY_PATH="/shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/lib/kblas/omp:${LD_LIBRARY_PATH}"
numactl --cpunodebind=0-3 --membind=0-3 ./winograd conf/final.conf
