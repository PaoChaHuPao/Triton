CCCL=/usr/local/cccl
CUB=$(CCCL)/cub
THRUST=$(CCCL)/thrust
LIBCUDA=$(CCCL)/libcudacxx/include
COMMON=$(PWD)/../

all: add.cu Makefile
	nvcc -o add add.cu -I$(THRUST) -I$(LIBCUDA) -I$(CUB) -I$(COMMON) -std=c++17 -lcurand -use_fast_math -lineinfo -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=\"compute_70,sm_70\"

clean:
	rm -f add

.PHONY: all clean
