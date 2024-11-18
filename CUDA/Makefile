#makefile to generate exe files for both CUDA and CPP versions
CC 			= gcc
CFLAGS 		= -Wall -g
NVCC 		= nvcc
CUDA_FLAGS 	= -gencode arch=compute_75,code=sm_75 -g

cuda : mandelbrot.cu
	$(NVCC) $(CUDA_FLAGS) $(XFLAG) mandelbrot.cu -o cuda_mandel

cpp : mandelbrot.cpp
	$(CC) $(CFLAGS) mandelbrot.cpp -o cpp_mandel

all : cuda cpp

clean:
	rm -rf *.o *.ppm *.pdb *.lib *.exp cpp_mandel cuda_mandel 

clean_cuda:
	rm -rf cuda*.o cuda*.ppm cuda*.pdb cuda*.lib cuda*.exp cuda_mandel

clean_cpp:
	rm -rf *.o cpp*.ppm cpp_mandel 
