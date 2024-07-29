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