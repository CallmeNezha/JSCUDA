
#ifndef __CUDA_COMMON_CUH__
#define __CUDA_COMMON_CUH__


#ifdef  JSCUDA_DLL_EXPORT
#define JSCUDA_DLL_API  __declspec(dllexport)
#else
#define JSCUDA_DLL_API  __declspec(dllimport)
#endif

#define JC_API_DEPRECATED  __declspec(deprecated("** This is a deprecated function **"))

namespace jc_cuda
{

extern "C"
{
    typedef enum 
    {
        JC_SUCCESS = 0,
        JC_PARAM_ERROR,
        JC_ERROR_COUNT
    } ErrorType;

    typedef enum {
        CUBLAS_ATOMICS_NOT_ALLOWED = 0,
        CUBLAS_ATOMICS_ALLOWED = 1
    } cublasAtomicsMode_t;

    typedef struct
    {
        unsigned int numRow;
        unsigned int numCol;
        float*       elements;
        bool         transposed;
        // In column major
    } Matrix;

    typedef struct
    {
        unsigned int numRow;
        unsigned int numCol;
        float**      elementsArray;
        bool         transposed;
        unsigned int count;
        // In column major
    } MatrixBatch;

    typedef struct
    {
        unsigned int length;
        float*       elements;
    } Vector;

    typedef struct cublasContext *  cublasHandle_t;
    typedef struct CUstream_st *    cudaStream_t;
    
    // cudaHelper functions
    JSCUDA_DLL_API bool      cudaDeviceInit(int argc, char **argv);
    JSCUDA_DLL_API void      cudaDeviceReset_t();
    JSCUDA_DLL_API void      cudaMalloc_t(void **devPtr, size_t size);
    JSCUDA_DLL_API void      cudaFree_t(void *devPtr);
    JSCUDA_DLL_API void      cudaSync();
    JSCUDA_DLL_API void      cudaMemcpyHostToDevice_t(const void *host, void *device, int offset_h, int offset_d, size_t size);
    JSCUDA_DLL_API void      cudaMemcpyDeviceToDevice_t(void *dst, const void *src, int offset_d, int offset_s, size_t size);
    JSCUDA_DLL_API void      cudaMemcpyDeviceToHost_t(const void *device, void *host, int offset_d, int offset_h, size_t size);
    
    
    // cuBLAS helper functions
    JSCUDA_DLL_API void      cublasCreate_t(cublasHandle_t *handle);
    JSCUDA_DLL_API void      cublasDestroy_t(cublasHandle_t handle);
    JSCUDA_DLL_API int       cublasGetVersion_t(cublasHandle_t handle);

    JSCUDA_DLL_API void      cublasSetStream_t(cublasHandle_t handle, cudaStream_t streamId);
    JSCUDA_DLL_API void      cublasGetStream_t(cublasHandle_t handle, cudaStream_t *streamId);

    JSCUDA_DLL_API void      cublasSetVector_t(int n, int elemSize, const void *x, int incx, void *y, int incy);
    JSCUDA_DLL_API void      cublasGetVector_t(int n, int elemSize, const void *x, int incx, void *y, int incy);

    JSCUDA_DLL_API void      cublasSetMatrix_t(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);
    JSCUDA_DLL_API void      cublasGetMatrix_t(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

    JSCUDA_DLL_API void      cublasSetVectorAsync_t(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream);
    JSCUDA_DLL_API void      cublasGetVectorAsync_t(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream);

    JSCUDA_DLL_API void      cublasSetMatrixAsync_t(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream);
    JSCUDA_DLL_API void      cublasGetMatrixAsync_t(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream);
    
    JSCUDA_DLL_API void      cublasSetAtomicsMode_t(cublasHandle_t handle, cublasAtomicsMode_t mode);
    JSCUDA_DLL_API void      cublasGetAtomicsMode_t(cublasHandle_t handle, cublasAtomicsMode_t *mode);

    // Math functions
    // Vectors related

    JSCUDA_DLL_API ErrorType vectorAdd(cublasHandle_t handle, Vector& vAd, const Vector& vBd);  // vAd += vBd
    JSCUDA_DLL_API ErrorType vectorCopy(const cublasHandle_t handle, Vector& vAd, const Vector& vBd); // vAd = vBd
    JSCUDA_DLL_API ErrorType vectorSwap(const cublasHandle_t handle, Vector& vAd, Vector& vBd);

    JSCUDA_DLL_API ErrorType vectorDot(const cublasHandle_t handle, const Vector& vAd, const Vector& vBd, float* resulth);
    JSCUDA_DLL_API ErrorType vectorNorm(const cublasHandle_t handle, const Vector& vd, float* resulth);
    JSCUDA_DLL_API ErrorType vectorMulScalar(const cublasHandle_t handle, Vector& vd, const float* scalarh);
    JSCUDA_DLL_API ErrorType vectorRank(const cublasHandle_t handle, const Vector& vAd, const Vector& vBd, Matrix& matd);
    

    // Matrix related
    JSCUDA_DLL_API ErrorType matrixMulScalar(const cublasHandle_t handle, Matrix& matd, const float* scalarh);
    JSCUDA_DLL_API ErrorType matrixMulVector(const cublasHandle_t handle, const Matrix& matAd, const Vector& vAd, Vector& vBd);
    JSCUDA_DLL_API ErrorType matrixMulMatrix(const cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd);
    JSCUDA_DLL_API ErrorType matrixMulMatrixBatched(const cublasHandle_t handle, const MatrixBatch& matAd, const MatrixBatch& matBd, MatrixBatch& matCd);
    /*JSCUDA_DLL_API ErrorType matrixEigen(cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd);
    JSCUDA_DLL_API ErrorType matrixUTri(cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd);
    JSCUDA_DLL_API ErrorType matrixDTri(cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd);
    JSCUDA_DLL_API ErrorType matrixLUD(cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd);
    JSCUDA_DLL_API ErrorType matrixSVD(cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd);
    JSCUDA_DLL_API ErrorType matrixQRD(cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd);
    JSCUDA_DLL_API ErrorType matrixCholeskyD(cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd);
    JSCUDA_DLL_API ErrorType matrixCGrad(cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd);*/


}
}


#endif // !__CUDA_COMMON_CUH__
