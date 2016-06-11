#define  JSCUDA_DLL_EXPORT

#include "cuda_common_kernel.cuh"
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

    JSCUDA_DLL_API 
    void cudaDeviceInit(int argc, char **argv)
    {
        int devID;
        // use command - line specified CUDA device, otherwise use device with highest Gflops / s
        devID = findCudaDevice(argc, (const char **)argv);
        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
        U_NUM_BLOCKSIZE = (deviceProp.major < 2) ? 16 : 32;
        U_NUM_THREAD = U_NUM_BLOCKSIZE * U_NUM_BLOCKSIZE;
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
    void cudaMemcpyHostToDevice_t(const void *host, void *device, int offset, size_t size)
    {
        checkCudaErrors(cudaMemcpy((char *)device + offset, host, size, cudaMemcpyHostToDevice));
    }
    JSCUDA_DLL_API
    void cudaMemcpyDeviceToDevice_t(void *dst, const void *src, int offset, size_t size)
    {
        checkCudaErrors(cudaMemcpy((char *)dst + offset, src, size, cudaMemcpyDeviceToDevice));
    }
    JSCUDA_DLL_API
    void cudaMemcpyDeviceToHost_t(const void *device, void *host, int offset, size_t size)
    {
        checkCudaErrors(cudaMemcpy((char *)host + offset, device, size, cudaMemcpyDeviceToHost));
    }

    //Round a / b to nearest higher integer value
    inline uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }
    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }
    JSCUDA_DLL_API
    void vectorAdd(const float* v1d, const float* v2d, float* vout, uint size)
    {
        uint numBlocks, numThreads;
        computeGridSize(size, U_NUM_THREAD, numBlocks, numThreads);
        vectorAddD << <numBlocks, numThreads >> >(v1d, v2d, vout, size);
        getLastCudaError("Kernel execution failed");
    }

    // global memory write operation in queue maybe not faster than CPU method
    JSCUDA_DLL_API
    float vectorInnerProduct(const float* v1d, const float* v2d, uint size)
    {
        float vout = 0.f;
        for (uint i = 0; i < size; i++)
        {
            vout += v1d[i] * v2d[i];
        }
        return vout;
    }

    JSCUDA_DLL_API
    void cublasCreate_t(cublasHandle_t *handle)
    {
        checkCudaErrors(cublasCreate(handle));
    }

    JSCUDA_DLL_API
    void cublasDestroy_t(cublasHandle_t handle)
    {
        checkCudaErrors(cublasDestroy(handle));
    }



    JSCUDA_DLL_API
    ErrorType matrixMulMatrix_blas(const cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matOd)
    {
        if (matAd.numCol != matBd.numRow) return JC_PARAM_ERROR;
        if (matAd.numRow != matOd.numRow || matBd.numCol != matOd.numCol) return JC_PARAM_ERROR;

        dim3 threads(U_NUM_BLOCKSIZE, U_NUM_BLOCKSIZE);
        dim3 grid(ceilfuint((float)matOd.numCol / threads.x), ceilfuint((float)matOd.numRow / threads.y));
        const float alpha = 1.f;
        const float beta = 0.f;
        checkCudaErrors(cublasSgemm(handle
            , CUBLAS_OP_N
            , CUBLAS_OP_N
            , matAd.numRow
            , matBd.numCol
            , matAd.numCol
            , &alpha
            , matAd.elements
            , matAd.numRow
            , matBd.elements
            , matBd.numRow
            , &beta
            , matOd.elements
            , matOd.numRow));

        getLastCudaError("Kernel execution failed");
        return JC_SUCCESS;
    }
}