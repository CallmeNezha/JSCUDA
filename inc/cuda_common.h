
#ifndef __CUDA_COMMON_CUH__
#define __CUDA_COMMON_CUH__


#ifdef  JSCUDA_DLL_EXPORT
#define JSCUDA_DLL_API  __declspec(dllexport)
#else
#define JSCUDA_DLL_API  __declspec(dllimport)
#endif
namespace jc_cuda
{

extern "C"
{
    enum ErrorType
    {
        JC_SUCCESS = 0,
        JC_PARAM_ERROR,
        JC_ERROR_COUNT
    };

    typedef struct
    {
        unsigned int numRow;
        unsigned int numCol;
        float*       elements;
        // In column major
    } Matrix;

    typedef struct cublasContext *  cublasHandle_t;

    JSCUDA_DLL_API bool      cudaDeviceInit(int argc, char **argv);
    JSCUDA_DLL_API void      cudaDeviceReset_t();
    JSCUDA_DLL_API void      cudaMalloc_t(void **devPtr, size_t size);
    JSCUDA_DLL_API void      cudaFree_t(void *devPtr);
    JSCUDA_DLL_API void      cudaSync();
    JSCUDA_DLL_API void      cudaMemcpyHostToDevice_t(const void *host, void *device, int offset, size_t size);
    JSCUDA_DLL_API void      cudaMemcpyDeviceToDevice_t(void *dst, const void *src, int offset, size_t size);
    JSCUDA_DLL_API void      cudaMemcpyDeviceToHost_t(const void *device, void *host, int offset, size_t size);
    JSCUDA_DLL_API void      vectorAdd(const float* v1d, const float* v2d, float* vout, unsigned int size);
    JSCUDA_DLL_API float     vectorInnerProduct(const float* v1d, const float* v2d, unsigned int size);
    JSCUDA_DLL_API void      cublasCreate_t(cublasHandle_t *handle);
    JSCUDA_DLL_API void      cublasDestroy_t(cublasHandle_t handle);
    JSCUDA_DLL_API ErrorType matrixMulMatrix_blas(cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matOd);

}
}


#endif // !__CUDA_COMMON_CUH__
