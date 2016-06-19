#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>
#include "jc_type.h"


#define  JSCUDA_DLL_EXPORT
#ifdef  JSCUDA_DLL_EXPORT
#define JSCUDA_DLL_API  __declspec(dllexport)
#else
#define JSCUDA_DLL_API  __declspec(dllimport)
#endif

extern "C"
{
    extern unsigned int  U_NUM_THREAD ;
    extern unsigned int  U_NUM_BLOCKSIZE ;
    

    // cuBLAS Helper function
    // 2.3.1
    JSCUDA_DLL_API
        void cublasCreate_t(cublasHandle_t* handle)
    {
        checkCudaErrors(cublasCreate(handle));
    }
    // 2.3.2
    JSCUDA_DLL_API
        void cublasDestroy_t(cublasHandle_t handle)
    {
        checkCudaErrors(cublasDestroy(handle));
    }
    // 2.3.3
    JSCUDA_DLL_API
        int cublasGetVersion_t(cublasHandle_t handle)
    {
        int version = 0;
        checkCudaErrors(cublasGetVersion(handle, &version));
        return version;
    }
    // 2.3.4
    JSCUDA_DLL_API
        void cublasSetStream_t(cublasHandle_t handle, cudaStream_t streamId)
    {
        checkCudaErrors(cublasSetStream(handle, streamId));
    }

    // 2.3.5
    JSCUDA_DLL_API
        void cublasGetStream_t(cublasHandle_t handle, cudaStream_t *streamId)
    {
        checkCudaErrors(cublasGetStream(handle, streamId));
    }
    //// 2.3.6 not yet
    //JSCUDA_DLL_API
    //    void cublasGetPointerMode_t(cublasHandle_t handle, cublasPointerMode_t *mode)
    //{
    //    checkCudaErrors(cublasGetPointerMode(handle, mode));
    //}

    //// 2.3.7 not yet
    //JSCUDA_DLL_API
    //    void cublasSetPointerMode_t(cublasHandle_t handle, cublasPointerMode_t mode)
    //{
    //    checkCudaErrors(cublasSetPointerMode(handle, mode));
    //}

    // 2.3.8
    JSCUDA_DLL_API
        void cublasSetVector_t(int n, int elemSize, const void *x, int incx, void *y, int incy)
    {
        checkCudaErrors(cublasSetVector(n, elemSize, x, incx, y, incy));
    }

    // 2.3.9
    JSCUDA_DLL_API
        void cublasGetVector_t(int n, int elemSize, const void *x, int incx, void *y, int incy)
    {
        checkCudaErrors(cublasGetVector(n, elemSize, x, incx, y, incy));
    }

    // 2.3.10
    JSCUDA_DLL_API
        void cublasSetMatrix_t(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)
    {
        checkCudaErrors(cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb));
    }
    
    // 2.3.11
    JSCUDA_DLL_API
        void cublasGetMatrix_t(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)
    {
        checkCudaErrors(cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb));
    }

    // 2.3.12
    JSCUDA_DLL_API
        void cublasSetVectorAsync_t(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream)
    {
        checkCudaErrors(cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream));
    }

    // 2.3.13
    JSCUDA_DLL_API
        void cublasGetVectorAsync_t(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream)
    {
        checkCudaErrors(cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream));
    }

    // 2.3.14
    JSCUDA_DLL_API
        void cublasSetMatrixAsync_t(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream)
    {
        checkCudaErrors(cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream));
    }

    // 2.3.15
    JSCUDA_DLL_API
        void cublasGetMatrixAsync_t(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream)
    {
        checkCudaErrors(cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream));
    }
    // 2.3.16
    JSCUDA_DLL_API
        void cublasSetAtomicsMode_t(cublasHandle_t handle, cublasAtomicsMode_t mode)
    {
        checkCudaErrors(cublasSetAtomicsMode(handle, mode));
    }
    // 2.3.17
    JSCUDA_DLL_API
        void cublasGetAtomicsMode_t(cublasHandle_t handle, cublasAtomicsMode_t *mode)
    {
        checkCudaErrors(cublasGetAtomicsMode(handle, mode));
    }

    // Vector functions
    JSCUDA_DLL_API
        ErrorType vectorCopy(const cublasHandle_t handle, Vector& vAd, const Vector& vBd)
    {
        if (vAd.length != vBd.length) return JC_PARAM_ERROR;
        if (vAd.length < 1 || vBd.length < 1) return JC_PARAM_ERROR;
        checkCudaErrors(cublasScopy(handle
            , vAd.length
            , vBd.elements
            , 1
            , vAd.elements
            , 1
            ));
        return JC_SUCCESS;
    }

    JSCUDA_DLL_API
        ErrorType vectorSwap(const cublasHandle_t handle, Vector& vAd, Vector& vBd)
    {
        if (vAd.length != vBd.length) return JC_PARAM_ERROR;
        if (vAd.length < 1 || vBd.length < 1) return JC_PARAM_ERROR;
        checkCudaErrors(cublasSswap(handle
            , vAd.length
            , vAd.elements
            , 1
            , vBd.elements
            , 1
            ));
        return JC_SUCCESS;
    }

    JSCUDA_DLL_API // vAd += vBd
        ErrorType vectorAdd(cublasHandle_t handle, Vector& vAd, const Vector& vBd)
    {
        if (vAd.length != vBd.length) return JC_PARAM_ERROR;
        if (vAd.length < 1 || vBd.length < 1) return JC_PARAM_ERROR;
        const float alpha = 1.f;
        checkCudaErrors(cublasSaxpy(handle
            , vAd.length
            , &alpha
            , vBd.elements
            , 1
            , vAd.elements
            , 1
            ));
        return JC_SUCCESS;
    }

    JSCUDA_DLL_API
        ErrorType vectorDot(const cublasHandle_t handle, const Vector& vAd, const Vector& vBd, float* resulth)
    {
        if (vAd.length != vBd.length) return JC_PARAM_ERROR;
        if (vAd.length < 1 || vBd.length < 1) return JC_PARAM_ERROR;
        checkCudaErrors(cublasSdot(handle
            , vAd.length
            , vAd.elements
            , 1
            , vBd.elements
            , 1
            , resulth
            ));
        return JC_SUCCESS;
    }

    JSCUDA_DLL_API
        ErrorType vectorNorm(const cublasHandle_t handle, const Vector& vd, float* resulth)
    {
        if (vd.length < 1) return JC_PARAM_ERROR;
        checkCudaErrors(cublasSnrm2(handle
            , vd.length
            , vd.elements
            , 1
            , resulth
            ));
        return JC_SUCCESS;
    }

    JSCUDA_DLL_API
        ErrorType vectorMulScalar(const cublasHandle_t handle, Vector& vd, const float* scalarh)
    {
        if (vd.length < 1) return JC_PARAM_ERROR;
        checkCudaErrors(cublasSscal(handle
            , vd.length
            , scalarh
            , vd.elements
            , 1
            ));
        return JC_SUCCESS;
    }

    // mat = vA * vB^T
    JSCUDA_DLL_API
        ErrorType vectorRank(const cublasHandle_t handle, const Vector& vAd, const Vector& vBd, Matrix& matd)
    {
        if (vAd.length != matd.numRow || vBd.length != matd.numCol) return JC_PARAM_ERROR;
        const float alpha = 1.f;
        checkCudaErrors(cublasSger(handle
            , matd.numRow
            , matd.numCol
            , &alpha
            , vAd.elements
            , 1
            , vBd.elements
            , 1
            , matd.elements
            , matd.numRow
            ));

        matd.transposed = false;
        return JC_SUCCESS;
    }


    // Vector - Matrix functions
    JSCUDA_DLL_API
        ErrorType matrixMulVector(const cublasHandle_t handle, const Matrix& matAd, const Vector& vAd, Vector& vBd)
    {
        cublasOperation_t opt;
        if (matAd.transposed)
        {
            if (matAd.numCol != vBd.length || matAd.numRow != vAd.length) return JC_PARAM_ERROR;
            opt = CUBLAS_OP_T;
        }
        else
        {
            if (matAd.numRow != vBd.length || matAd.numCol != vAd.length) return JC_PARAM_ERROR;
            opt = CUBLAS_OP_N;
        }
        const float alpha = 1.f;
        const float beta = 0.f;
        unsigned int lda = matAd.numRow;

        checkCudaErrors(cublasSgemv(handle
            , opt
            , matAd.numRow
            , matAd.numCol
            , &alpha
            , matAd.elements
            , lda
            , vAd.elements
            , 1
            , &beta
            , vBd.elements
            , 1
            ));

        return JC_SUCCESS;
    }


    // Matrix functions
    JSCUDA_DLL_API
        ErrorType matrixMulScalar(const cublasHandle_t handle, Matrix& matd, const float* scalarh)
    {
        if (matd.numCol < 1 || matd.numRow < 1) return JC_PARAM_ERROR;
        Vector v{ matd.numRow * matd.numCol, matd.elements};
        return vectorMulScalar(handle, v, scalarh);
    }

    JSCUDA_DLL_API
        ErrorType matrixMulMatrix(const cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd)
    {
        if (matAd.numCol < 1 || matAd.numRow < 1 ||
            matBd.numCol < 1 || matBd.numRow < 1 ||
            matCd.numCol < 1 || matCd.numRow < 1) return JC_PARAM_ERROR;

        
        cublasOperation_t optA, optB;
        unsigned int lda, ldb, ldc, k;
        lda = matAd.numRow; ldb = matBd.numRow;
        ldc = matCd.numRow;

        if (!matAd.transposed && !matBd.transposed)
        {
            if (matAd.numCol != matBd.numRow)         return JC_PARAM_ERROR;
            if (matAd.numRow != matCd.numRow || matBd.numCol != matCd.numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_N; optB = CUBLAS_OP_N;
            k = matAd.numCol;
        }
        else if (matAd.transposed && !matBd.transposed)
        {
            if (matAd.numRow != matBd.numRow)         return JC_PARAM_ERROR;
            if (matAd.numCol != matCd.numRow || matBd.numCol != matCd.numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_T; optB = CUBLAS_OP_N;
            k = matAd.numRow;
        }
        else if (!matAd.transposed && matBd.transposed)
        {
            if (matAd.numCol != matBd.numCol)         return JC_PARAM_ERROR;
            if (matAd.numRow != matCd.numRow || matBd.numRow != matCd.numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_N; optB = CUBLAS_OP_T;
            k = matAd.numCol;
        }
        else if (matAd.transposed && matBd.transposed)
        {
            if (matAd.numRow != matBd.numCol)         return JC_PARAM_ERROR;
            if (matAd.numCol != matCd.numRow || matBd.numRow != matCd.numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_T; optB = CUBLAS_OP_T;
            k = matAd.numRow;
        }

        const float alpha = 1.f;
        const float beta  = 0.f;

        checkCudaErrors(cublasSgemm(handle
            , optA
            , optB
            , matCd.numRow
            , matCd.numCol
            , k
            , &alpha
            , matAd.elements
            , lda
            , matBd.elements
            , ldb
            , &beta
            , matCd.elements
            , ldc
            ));
        
        matCd.transposed = false;
        return JC_SUCCESS;
    }

    
    JSCUDA_DLL_API
        ErrorType matrixMulMatrixBatched(const cublasHandle_t handle, const MatrixBatch& matAd, const MatrixBatch& matBd, MatrixBatch& matCd)
    {
        if (matAd.numCol < 1 || matAd.numRow < 1 ||
            matBd.numCol < 1 || matBd.numRow < 1 ||
            matCd.numCol < 1 || matCd.numRow < 1) return JC_PARAM_ERROR;


        cublasOperation_t optA, optB;
        unsigned int lda, ldb, ldc, k;
        lda = matAd.numRow; ldb = matBd.numRow;
        ldc = matCd.numRow;

        if (!matAd.transposed && !matBd.transposed)
        {
            if (matAd.numCol != matBd.numRow)         return JC_PARAM_ERROR;
            if (matAd.numRow != matCd.numRow || matBd.numCol != matCd.numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_N; optB = CUBLAS_OP_N;
            k = matAd.numCol;
        }
        else if (matAd.transposed && !matBd.transposed)
        {
            if (matAd.numRow != matBd.numRow)         return JC_PARAM_ERROR;
            if (matAd.numCol != matCd.numRow || matBd.numCol != matCd.numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_T; optB = CUBLAS_OP_N;
            k = matAd.numRow;
        }
        else if (!matAd.transposed && matBd.transposed)
        {
            if (matAd.numCol != matBd.numCol)         return JC_PARAM_ERROR;
            if (matAd.numRow != matCd.numRow || matBd.numRow != matCd.numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_N; optB = CUBLAS_OP_T;
            k = matAd.numCol;
        }
        else if (matAd.transposed && matBd.transposed)
        {
            if (matAd.numRow != matBd.numCol)         return JC_PARAM_ERROR;
            if (matAd.numCol != matCd.numRow || matBd.numRow != matCd.numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_T; optB = CUBLAS_OP_T;
            k = matAd.numRow;
        }

        const float alpha = 1.f;
        const float beta = 0.f;
        
        checkCudaErrors(cublasSgemmBatched(handle
            , optA
            , optB
            , matCd.numRow
            , matCd.numCol
            , k
            , &alpha
            , (const float**)matAd.elementsArray
            , lda
            , (const float**)matBd.elementsArray
            , ldb
            , &beta
            , matCd.elementsArray
            , ldc
            , matAd.count
            ));

        matCd.transposed = false;
        return JC_SUCCESS;
    }


    

}