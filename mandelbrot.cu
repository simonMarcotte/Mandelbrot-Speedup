#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t result_ = (call); \
    if (result_ != cudaSuccess) { \
        fprintf(stderr, "%s:%d: CUDA error %d: %s\n", __FILE__, __LINE__, result_, cudaGetErrorString(result_)); \
        exit(1); \
    } \
} while (0)

__device__ int mandelbrot(double x, double y, int maxiter) {
    double u = 0.0;
    double v = 0.0;
    double u2 = u * u;
    double v2 = v * v;
    int k;
    for (k = 1; k < maxiter && (u2 + v2 < 4.0); k++) {
        v = 2 * u * v + y;
        u = u2 - v2 + x;
        u2 = u * u;
        v2 = v * v;
    }
    return k;
}

__global__ void mandelbrotKernel(double xmin, double ymin, double dx, double dy, int maxiter, int xres, int yres, unsigned char* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < xres && j < yres) {
        double x = xmin + i * dx;
        double y = ymin + j * dy;
        int k = mandelbrot(x, y, maxiter);
        int index = (j * xres + i) * 6;
        if (k >= maxiter) {
            // Interior
            result[index] = 0;
            result[index + 1] = 0;
            result[index + 2] = 0;
            result[index + 3] = 0;
            result[index + 4] = 0;
            result[index + 5] = 0;
        } else {
            // Exterior
            result[index] = k >> 8;
            result[index + 1] = k & 255;
            result[index + 2] = k >> 8;
            result[index + 3] = k & 255;
            result[index + 4] = k >> 8;
            result[index + 5] = k & 255;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 8) {
        printf("Usage:   %s <xmin> <xmax> <ymin> <ymax> <maxiter> <xres> <out.ppm>\n", argv[0]);
        printf("test");
        printf("Example: %s 0.27085 0.27100 0.004640 0.004810 1000 1024 pic.ppm\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    time_t start, end;

    time(&start);

    const double xmin = atof(argv[1]);
    const double xmax = atof(argv[2]);
    const double ymin = atof(argv[3]);
    const double ymax = atof(argv[4]);
    const uint16_t maxiter = (unsigned short)atoi(argv[5]);
    const int xres = atoi(argv[6]);
    const int yres = (xres * (ymax - ymin)) / (xmax - xmin);
    const char* filename = argv[7];

    FILE* fp = fopen(filename, "wb");
    char* comment = "# Mandelbrot set";
    fprintf(fp,
            "P6\n# Mandelbrot, xmin=%lf, xmax=%lf, ymin=%lf, ymax=%lf, maxiter=%d\n%d\n%d\n%d\n",
            xmin, xmax, ymin, ymax, maxiter, xres, yres, (maxiter < 256 ? 256 : maxiter));

    double dx = (xmax - xmin) / xres;
    double dy = (ymax - ymin) / yres;

    unsigned char* result;
    CUDA_CHECK(cudaMallocManaged(&result, xres * yres * 6 * sizeof(unsigned char)));

    dim3 blockSize(16, 16);
    dim3 gridSize((xres + blockSize.x - 1) / blockSize.x, (yres + blockSize.y - 1) / blockSize.y);
    mandelbrotKernel<<<gridSize, blockSize>>>(xmin, ymin, dx, dy, maxiter, xres, yres, result);
    CUDA_CHECK(cudaDeviceSynchronize());

    fwrite(result, xres * yres * 6, 1, fp);
    fclose(fp);
    cudaFree(result);

    time(&end);
    double time_taken = double(end - start);
    
    printf("Generating Cuda Mandelbrot took: %.5f s", time_taken);

    return 0;
}
