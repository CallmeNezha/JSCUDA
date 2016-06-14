#define  JSCUDA_DLL_EXPORT

#include <cuda_runtime.h>
#include <helper_math.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#ifdef  JSCUDA_DLL_EXPORT
#define JSCUDA_DLL_API  __declspec(dllexport)
#else
#define JSCUDA_DLL_API  __declspec(dllimport)
#endif

extern "C"
{
    //Default parameter, may scalar by 2 for Fermi or above
    unsigned int  U_NUM_THREAD = 256;
    unsigned int  U_NUM_BLOCKSIZE = 16;

    JSCUDA_DLL_API 
    bool cudaDeviceInit(int argc, char **argv)
    {
        int devID;
        // use command - line specified CUDA device, otherwise use device with highest Gflops / s
        devID = findCudaDevice(argc, (const char **)argv);
        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            return false;
        }
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
        U_NUM_BLOCKSIZE = (deviceProp.major < 2) ? 16 : 32;
        U_NUM_THREAD = U_NUM_BLOCKSIZE * U_NUM_BLOCKSIZE;
        return true;
    }
    JSCUDA_DLL_API
    void cudaDeviceReset_t()
    {
        checkCudaErrors(cudaDeviceReset());
    }
    JSCUDA_DLL_API
    void cudaMalloc_t(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }
    JSCUDA_DLL_API
    void cudaFree_t(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }
    JSCUDA_DLL_API
    void cudaSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }
    JSCUDA_DLL_API
    void cudaMemcpyHostToDevice_t(const void *host, void *device, int offset_h, int offset_d, size_t size)
    {
        checkCudaErrors(cudaMemcpy((char *)device + offset_d, (char *)host + offset_h, size, cudaMemcpyHostToDevice));
    }
    JSCUDA_DLL_API
    void cudaMemcpyDeviceToDevice_t(void *dst, const void *src, int offset_d, int offset_s, size_t size)
    {
        checkCudaErrors(cudaMemcpy((char *)dst + offset_d, (char *)src + offset_s, size, cudaMemcpyDeviceToDevice));
    }
    JSCUDA_DLL_API
    void cudaMemcpyDeviceToHost_t(const void *device, void *host, int offset_d, int offset_h, size_t size)
    {
        checkCudaErrors(cudaMemcpy((char *)host + offset_h, (char *)device + offset_d, size, cudaMemcpyDeviceToHost));
    }

    
}