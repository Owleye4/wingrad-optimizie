#!/bin/bash
#SBATCH -n 1
#SBATCH -o output/job%j.out
#SBATCH -e error/job%j.err
#SBATCH -c 64
#SBATCH --exclusive
export OMP_NUN_THREADS=64
numactl --cpunodebind=0-3 --membind=0-3 perf stat -ddd ./winograd conf/final.conf
