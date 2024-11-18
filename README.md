# CUDA/HIP-Mandelbrot-Speedup
Using AMD ROCm HIP and NVIDIA CUDA to speedup the rendering of Mandelbrot fractals

Here is an example mandelbrot set I have generated:

![Basic Mandelbrot Set](Images/mandelbrot_example.png)


## Setting Up
This covers setting up AMD's ROCm and HIP SDK.

To get started, I installed the HIP SDK from AMD [here](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html)

I am personally using a Windows 11 Machine, running an AMD Radeon 7800 XT GPU. For the CUDA version, I ran Windows 11 with a NVIDIA RTX 2070 Super.

Once ROCm is installed (presumably in the `C:\AMD` directory), I created a `hipcc.bat` script to run the hipcc.exe exectuable anywhere on my computer:
```
@echo off
"C:\AMD\ROCm\6.1\bin\hipcc" %*
```

After this, running `hipcc --version` should produce:
```
> hipcc --version
HIP version: 6.1.40252-53f3e11ac
clang version 19.0.0git (git@github.amd.com:Compute-Mirrors/llvm-project b3dbdf4f03718d63a3292f784216fddb3e73d521).
Target: x86_64-pc-windows-msvc
Thread model: posix
InstalledDir: C:\AMD\ROCm\6.1\bin
```

When wanting to use `hipcc` to compile HIP code, make sure to target the correct LLVM architecture of the GPU currently being used. For me since I have an 7800 XT, i use `--amdgpu-target=gfx1101`.
See a list of LLVM targets [here](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html).

From here, I can use HIP to compile and run CUDA like program to parralelize kernel functions on an AMD GPU.

**TODO**:

Both programs can be optimized further based on GPU specifications. See the GPU_info folder to run a script to get GPU info to optimize threads and blocks further.


## Usage

To run either the CUDA or HIP versions of of generating a mandelbrot fractal use the following:

1. Run `make <language>`, where `language` is `cuda` or `hip`, in the corresponding directory
2. Run `./<executable> <xmin> <xmax> <ymin> <ymax> <maxiter> <xres> <out.ppm>`
    - For example: `.\hip_mandel.exe -1.7 0.7 -1.0 1.0 1000 10240 output.ppm`
    - `<executable>` specifies the executable being used, either from CUDA, HIP, or C++.
    - `<xmin>` and `<xmax>`correspond to the minimum and maximum real values on the imaginary plane.
    - `<ymin>` and `<ymax>`correspond to the minimum and maximum imaginary values on the imaginary plane.
    - `<maxiter>` specifies the maximum iterations in computing mandelbrot set.
    - `<xres>` specifies the x-resolution of the image. y-resolution is determined dynamically to maintain aspect ratio.
    - `<out.ppm>` the .ppm image file to generate.
3. View the output `.ppm` image in a `.ppm` viewer of choice