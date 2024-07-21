#!/bin/bash
#SBATCH -p pac
#SBATCH -n 1
#SBATCH -o output/job%j.out
#SBATCH -e error/job%j.err
#SBATCH -c 160
#SBATCH --exclusive
export OMP_NUM_THREADS=160
numactl --cpunodebind=0-3 --membind=0-3 perf stat -ddd ./winograd smallrealworld.conf 1
