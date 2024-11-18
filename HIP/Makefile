#makefile to generate exe files for both CUDA and CPP versions
CC 			= gcc
CFLAGS 		= -Wall -g
HIPCC 		= hipcc
HIP_FLAGS 	= --amdgpu-target=gfx1101

hip : hip_mandelbrot.cu
	$(HIPCC) $(HIP_FLAGS) $(XFLAG) hip_mandelbrot.cu -o hip_mandel.exe

cpp : mandelbrot.cpp
	$(CC) $(CFLAGS) mandelbrot.cpp -o cpp_mandel

all : hip cpp

clean:
	rm -rf *.o *.ppm *.pdb *.lib *.exp cpp_mandel hip_mandel.exe 

clean_cuda:
	rm -rf cuda*.o cuda*.ppm cuda*.pdb cuda*.lib cuda*.exp cuda_mandel

clean_cpp:
	rm -rf *.o cpp*.ppm cpp_mandel 
