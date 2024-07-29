# 目录结构
```
.
├── common.h   # 源码
├── conf	   # 算例文件
│   ├── final.conf
│   └── vgg16.conf
├── driver.c
├── env.sh	   # 环境配置文件
├── Makefile   # 编译文件
├── README.md  
├── run.sh     # 运行脚本
└── winograd.c # 源码
```
# 如何运行
## 准备环境
```Bash
source env.sh  // 激活module环境
```
## 编译
```Bash
make
```
## 运行
### 在本地运行
```Bash
chmod u+x run.sh
./run.sh
```
### 通过作业调度系统运行
```Bash
sbatch run.sh
```

### 选择验证模式
只需要将run.sh中的第10行
```bash 
numactl --cpunodebind=0-3 --membind=0-3 ./winograd conf/final.conf
```
修改为
```bash
numactl --cpunodebind=0-3 --membind=0-3 ./winograd conf/final.conf 1
```
即可验证正确性。

# 优化结果
技术报告中的结果为连续运行十次的平均值，以下为连续运行十次的结果。
```bash
[PAC20233407@c11n089 winograd]$ ./run.sh 
Layer 0 :  Elapse time 8.400000 ms. (   36.91 GFlops) 
Layer 1 :  Elapse time 55.750000 ms. (  830.38 GFlops) 
Layer 2 :  Elapse time 468.933667 ms. ( 2505.54 GFlops) 
Layer 3 :  Elapse time 98.633333 ms. (  465.32 GFlops) 
Layer 4 :  Elapse time 247.844667 ms. ( 4721.63 GFlops) 
Layer 5 :  Elapse time 634.391667 ms. ( 7378.59 GFlops) 
Layer 6 :  Elapse time 29.783667 ms. ( 5840.32 GFlops) 
Layer 7 :  Elapse time 578.255667 ms. ( 7495.23 GFlops) 
Total elapse time: 2.121993. ( 5479.14 GFlops) 
[PAC20233407@c11n089 winograd]$ ./run.sh 
Layer 0 :  Elapse time 7.755333 ms. (   39.97 GFlops) 
Layer 1 :  Elapse time 70.226000 ms. (  659.21 GFlops) 
Layer 2 :  Elapse time 466.697333 ms. ( 2517.55 GFlops) 
Layer 3 :  Elapse time 91.455333 ms. (  501.84 GFlops) 
Layer 4 :  Elapse time 264.310333 ms. ( 4427.48 GFlops) 
Layer 5 :  Elapse time 587.610667 ms. ( 7966.02 GFlops) 
Layer 6 :  Elapse time 30.529000 ms. ( 5697.74 GFlops) 
Layer 7 :  Elapse time 592.373000 ms. ( 7316.60 GFlops) 
Total elapse time: 2.110957. ( 5507.78 GFlops) 
[PAC20233407@c11n089 winograd]$ ./run.sh 
Layer 0 :  Elapse time 14.095333 ms. (   21.99 GFlops) 
Layer 1 :  Elapse time 94.762000 ms. (  488.53 GFlops) 
Layer 2 :  Elapse time 500.072000 ms. ( 2349.53 GFlops) 
Layer 3 :  Elapse time 91.703667 ms. (  500.48 GFlops) 
Layer 4 :  Elapse time 261.318000 ms. ( 4478.18 GFlops) 
Layer 5 :  Elapse time 626.787333 ms. ( 7468.11 GFlops) 
Layer 6 :  Elapse time 40.165333 ms. ( 4330.75 GFlops) 
Layer 7 :  Elapse time 558.616333 ms. ( 7758.74 GFlops) 
Total elapse time: 2.187520. ( 5315.01 GFlops) 
[PAC20233407@c11n089 winograd]$ ./run.sh 
Layer 0 :  Elapse time 9.651667 ms. (   32.12 GFlops) 
Layer 1 :  Elapse time 52.800000 ms. (  876.78 GFlops) 
Layer 2 :  Elapse time 492.921000 ms. ( 2383.62 GFlops) 
Layer 3 :  Elapse time 90.251000 ms. (  508.54 GFlops) 
Layer 4 :  Elapse time 250.545667 ms. ( 4670.72 GFlops) 
Layer 5 :  Elapse time 556.895333 ms. ( 8405.38 GFlops) 
Layer 6 :  Elapse time 28.751000 ms. ( 6050.09 GFlops) 
Layer 7 :  Elapse time 548.071333 ms. ( 7908.02 GFlops) 
Total elapse time: 2.029887. ( 5727.75 GFlops) 
[PAC20233407@c11n089 winograd]$ ./run.sh 
Layer 0 :  Elapse time 8.246333 ms. (   37.59 GFlops) 
Layer 1 :  Elapse time 83.400667 ms. (  555.08 GFlops) 
Layer 2 :  Elapse time 477.971000 ms. ( 2458.17 GFlops) 
Layer 3 :  Elapse time 88.100333 ms. (  520.95 GFlops) 
Layer 4 :  Elapse time 242.678667 ms. ( 4822.14 GFlops) 
Layer 5 :  Elapse time 575.754667 ms. ( 8130.06 GFlops) 
Layer 6 :  Elapse time 29.513333 ms. ( 5893.82 GFlops) 
Layer 7 :  Elapse time 593.385333 ms. ( 7304.12 GFlops) 
Total elapse time: 2.099050. ( 5539.02 GFlops) 
[PAC20233407@c11n089 winograd]$ ./run.sh 
Layer 0 :  Elapse time 10.627667 ms. (   29.17 GFlops) 
Layer 1 :  Elapse time 114.103667 ms. (  405.72 GFlops) 
Layer 2 :  Elapse time 592.920667 ms. ( 1981.60 GFlops) 
Layer 3 :  Elapse time 92.735333 ms. (  494.91 GFlops) 
Layer 4 :  Elapse time 258.572667 ms. ( 4525.73 GFlops) 
Layer 5 :  Elapse time 562.648000 ms. ( 8319.44 GFlops) 
Layer 6 :  Elapse time 29.378667 ms. ( 5920.83 GFlops) 
Layer 7 :  Elapse time 536.591333 ms. ( 8077.21 GFlops) 
Total elapse time: 2.197578. ( 5290.68 GFlops) 
[PAC20233407@c11n089 winograd]$ ./run.sh 
Layer 0 :  Elapse time 9.296667 ms. (   33.35 GFlops) 
Layer 1 :  Elapse time 74.784667 ms. (  619.03 GFlops) 
Layer 2 :  Elapse time 465.290667 ms. ( 2525.16 GFlops) 
Layer 3 :  Elapse time 96.735000 ms. (  474.45 GFlops) 
Layer 4 :  Elapse time 247.449333 ms. ( 4729.17 GFlops) 
Layer 5 :  Elapse time 594.212667 ms. ( 7877.51 GFlops) 
Layer 6 :  Elapse time 29.293667 ms. ( 5938.01 GFlops) 
Layer 7 :  Elapse time 597.378667 ms. ( 7255.30 GFlops) 
Total elapse time: 2.114441. ( 5498.70 GFlops) 
[PAC20233407@c11n089 winograd]$ ./run.sh 
Layer 0 :  Elapse time 9.991333 ms. (   31.03 GFlops) 
Layer 1 :  Elapse time 70.079667 ms. (  660.59 GFlops) 
Layer 2 :  Elapse time 469.337000 ms. ( 2503.39 GFlops) 
Layer 3 :  Elapse time 89.032667 ms. (  515.49 GFlops) 
Layer 4 :  Elapse time 245.842333 ms. ( 4760.08 GFlops) 
Layer 5 :  Elapse time 601.268333 ms. ( 7785.07 GFlops) 
Layer 6 :  Elapse time 29.307667 ms. ( 5935.18 GFlops) 
Layer 7 :  Elapse time 597.849333 ms. ( 7249.58 GFlops) 
Total elapse time: 2.112708. ( 5503.21 GFlops) 
[PAC20233407@c11n089 winograd]$ ./run.sh 
Layer 0 :  Elapse time 10.892333 ms. (   28.46 GFlops) 
Layer 1 :  Elapse time 52.029667 ms. (  889.76 GFlops) 
Layer 2 :  Elapse time 472.791333 ms. ( 2485.10 GFlops) 
Layer 3 :  Elapse time 92.064333 ms. (  498.52 GFlops) 
Layer 4 :  Elapse time 262.518333 ms. ( 4457.71 GFlops) 
Layer 5 :  Elapse time 582.868000 ms. ( 8030.84 GFlops) 
Layer 6 :  Elapse time 32.248333 ms. ( 5393.96 GFlops) 
Layer 7 :  Elapse time 562.577667 ms. ( 7704.11 GFlops) 
Total elapse time: 2.067990. ( 5622.22 GFlops) 
[PAC20233407@c11n089 winograd]$ ./run.sh 
Layer 0 :  Elapse time 9.226667 ms. (   33.60 GFlops) 
Layer 1 :  Elapse time 76.535333 ms. (  604.87 GFlops) 
Layer 2 :  Elapse time 478.950667 ms. ( 2453.14 GFlops) 
Layer 3 :  Elapse time 98.269667 ms. (  467.04 GFlops) 
Layer 4 :  Elapse time 256.616000 ms. ( 4560.24 GFlops) 
Layer 5 :  Elapse time 595.650667 ms. ( 7858.50 GFlops) 
Layer 6 :  Elapse time 46.426000 ms. ( 3746.74 GFlops) 
Layer 7 :  Elapse time 568.384667 ms. ( 7625.40 GFlops) 
Total elapse time: 2.130060. ( 5458.39 GFlops) 
```
以下是正确性验证的结果
```bash
[PAC20233407@c11n087 Winograd-commit]$ ./run.sh
Layer 0 : (Channel Height Weight Filter Batch) = (3   1200 1200 2   2  ) : Validation Passed !
Layer 1 : (Channel Height Weight Filter Batch) = (7   1200 1200 32  8  ) : Validation Passed !
Layer 2 : (Channel Height Weight Filter Batch) = (8   1000 1000 512 16 ) : Validation Passed !
Layer 3 : (Channel Height Weight Filter Batch) = (16  1000 1000 5   32 ) : Validation Passed !
Layer 4 : (Channel Height Weight Filter Batch) = (32  500 500 128 64 ) : Validation Passed !
Layer 5 : (Channel Height Weight Filter Batch) = (64  500 500 256 64 ) : Validation Passed !
Layer 6 : (Channel Height Weight Filter Batch) = (256 50  50  256 64 ) : Validation Passed !
Layer 7 : (Channel Height Weight Filter Batch) = (256 50  600 512 64 ) : Validation Passed !
```