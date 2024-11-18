#include <stdio.h>
#include <hip/hip_runtime.h>


// Compile with hipcc -o device.exe ..\device.cu

int main() {
    hipDeviceProp_t prop;
    int deviceCount;

    // Get the number of devices
    hipGetDeviceCount(&deviceCount);
    printf("Number of GPUs: %d\n", deviceCount);

    for (int device = 0; device < deviceCount; device++) {
        hipGetDeviceProperties(&prop, device);

        printf("\nDevice %d: %s\n", device, prop.name);
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Max blocks per grid: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max threads per dimension (block): (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }

    return 0;
}
