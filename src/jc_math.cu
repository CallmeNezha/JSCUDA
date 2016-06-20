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

    ErrorType getMatrixBlasParam(unsigned int matA_numRow
        , unsigned int matA_numCol
        , bool matA_transposed
        , unsigned int matB_numRow
        , unsigned int matB_numCol
        , bool matB_transposed
        , unsigned int matC_numRow
        , unsigned int matC_numCol
        , cublasOperation_t& optA
        , cublasOperation_t& optB
        , unsigned int& lda
        , unsigned int& ldb
        , unsigned int& ldc
        , unsigned int& k)
    {
        if (matA_numCol < 1 || matA_numRow < 1 ||
            matB_numCol < 1 || matB_numRow < 1 ||
            matC_numCol < 1 || matC_numRow < 1) return JC_PARAM_ERROR;
        if (!matA_transposed && !matB_transposed)
        {
            if (matA_numCol != matB_numRow)         return JC_PARAM_ERROR;
            if (matA_numRow != matC_numRow || matB_numCol != matC_numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_N; optB = CUBLAS_OP_N;
            k = matA_numCol;
        }
        else if (matA_transposed && !matB_transposed)
        {
            if (matA_numRow != matB_numRow)         return JC_PARAM_ERROR;
            if (matA_numCol != matC_numRow || matB_numCol != matC_numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_T; optB = CUBLAS_OP_N;
            k = matA_numRow;
        }
        else if (!matA_transposed && matB_transposed)
        {
            if (matA_numCol != matB_numCol)         return JC_PARAM_ERROR;
            if (matA_numRow != matC_numRow || matB_numRow != matC_numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_N; optB = CUBLAS_OP_T;
            k = matA_numCol;
        }
        else if (matA_transposed && matB_transposed)
        {
            if (matA_numRow != matB_numCol)         return JC_PARAM_ERROR;
            if (matA_numCol != matC_numRow || matB_numRow != matC_numCol) return JC_PARAM_ERROR;
            optA = CUBLAS_OP_T; optB = CUBLAS_OP_T;
            k = matA_numRow;
        }
        lda = matA_numRow; 
        ldb = matB_numRow;
        ldc = matC_numRow;
        return JC_SUCCESS;
    }

    JSCUDA_DLL_API
        ErrorType matrixMulMatrix(const cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd)
    {
        cublasOperation_t optA, optB;
        unsigned int lda, ldb, ldc, k;

        ErrorType ret;
        ret = getMatrixBlasParam(matAd.numRow
            , matAd.numCol
            , matAd.transposed
            , matBd.numRow
            , matBd.numCol
            , matBd.transposed
            , matCd.numRow
            , matCd.numCol
            , optA
            , optB
            , lda
            , ldb
            , ldc
            , k
            );
        if (JC_SUCCESS != ret) return ret; // Check return

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
        cublasOperation_t optA, optB;
        unsigned int lda, ldb, ldc, k;

        ErrorType ret;
        ret = getMatrixBlasParam(matAd.numRow
            , matAd.numCol
            , matAd.transposed
            , matBd.numRow
            , matBd.numCol
            , matBd.transposed
            , matCd.numRow
            , matCd.numCol
            , optA
            , optB
            , lda
            , ldb
            , ldc
            , k
            );
        if (JC_SUCCESS != ret) return ret; // Check return

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

    // BLAS-like extension
    JSCUDA_DLL_API
        ErrorType matrixAdd(const cublasHandle_t handle, const Matrix& matAd, const Matrix& matBd, Matrix& matCd)
    {
        cublasOperation_t optA, optB;
        unsigned int lda, ldb, ldc, k;

        ErrorType ret;
        ret = getMatrixBlasParam(matAd.numRow
            , matAd.numCol
            , matAd.transposed
            , matBd.numRow
            , matBd.numCol
            , matBd.transposed
            , matCd.numRow
            , matCd.numCol
            , optA
            , optB
            , lda
            , ldb
            , ldc
            , k
            );
        if (JC_SUCCESS != ret) return ret; // Check return

        const float alpha = 1.f;
        const float beta = 1.f;

        checkCudaErrors(cublasSgeam(handle
            , optA
            , optB
            , matCd.numRow
            , matCd.numCol
            , &alpha
            , matAd.elements
            , lda
            , &beta
            , matBd.elements
            , ldb
            , matCd.elements
            , ldc
            ));

        matCd.transposed = false;
        return JC_SUCCESS;
    }

    JSCUDA_DLL_API
        ErrorType matrixGetrfBatched(const cublasHandle_t handle, MatrixBatch& matd, int* pivotArrayd, int* infoArrayd)
    {
        if (matd.numRow != matd.numCol) // TODO: Deal with tranposed matrix
            return JC_PARAM_ERROR;
        if (matd.transposed)
        {
            fprintf(stderr, "matrixGetrfBatched Not Support Transposed Matrices Yet.");
            return JC_PARAM_ERROR;
        }
        int n = matd.numRow;
        float ** Aarray = matd.elementsArray;
        int lda = matd.numRow;
        int batchSize = matd.count;

        checkCudaErrors(cublasSgetrfBatched(handle
            , n
            , Aarray
            , lda
            , pivotArrayd
            , infoArrayd
            , batchSize
            ));
        return JC_SUCCESS;
    }

    // matrixGetrsBatched_t
    // @param matBd in/out
    // @param pivotArrayd in   device
    // @param infoArrayh  out  host
    JSCUDA_DLL_API
        ErrorType matrixGetrsBatched(const cublasHandle_t handle, const MatrixBatch& matAd, MatrixBatch& matBd, int* pivotArrayd, int* infoArrayh)
    {
        if (matAd.count != matBd.count || matAd.numRow != matAd.numCol || matAd.numCol != matBd.numRow)
            return JC_PARAM_ERROR;
        if (matAd.transposed)
        {
            fprintf(stderr, "matrixGetrsBatched Not Support Transposed Matrices Yet.");
            return JC_PARAM_ERROR;
        }
        cublasOperation_t trans = CUBLAS_OP_N; // TODO: Deal with tranposed matrix
        int n = matAd.numRow;
        int nrhs = matBd.numCol;
        float ** Aarray = matAd.elementsArray;
        int lda = matAd.numRow;
        int *devIpiv = pivotArrayd;
        float ** Barray = matBd.elementsArray;
        int ldb = matBd.numRow;
        int *info = infoArrayh;
        int batchSize = matAd.count;
        checkCudaErrors(cublasSgetrsBatched(handle
            , trans
            , n
            , nrhs
            , (const float **)Aarray
            , lda
            , devIpiv
            , Barray
            , ldb
            , info
            , batchSize
            ));
        matBd.transposed = false;
        return JC_SUCCESS;
    }

    JSCUDA_DLL_API
        ErrorType matrixGetriBatched(const cublasHandle_t handle, const MatrixBatch& matAd, MatrixBatch& matBd, int* pivotArrayd, int* infoArrayd)
    {
        if (matAd.count != matBd.count || matAd.numRow != matAd.numCol || matAd.numCol != matBd.numRow || matBd.numRow != matBd.numCol)
            return JC_PARAM_ERROR;
        if (matAd.transposed)
        {
            fprintf(stderr, "matrixGetriBatched Not Support Transposed Matrices Yet.");
            return JC_PARAM_ERROR;
        }

        int n = matAd.numRow;
        float ** Aarray = matAd.elementsArray;
        int lda = matAd.numRow;
        float ** Barray = matBd.elementsArray;
        int ldb = matBd.numRow;
        int *info = infoArrayd;
        int batchSize = matAd.count;

        checkCudaErrors(cublasSgetriBatched(handle
            , n
            , (const float **)Aarray
            , lda
            , pivotArrayd
            , Barray
            , ldb
            , info
            , batchSize
            ));

        matBd.transposed = false;
        return JC_SUCCESS;
    }

    // This function only works when 'n' is less than 32
    // if not, use matrixGetrfBatched and matrixGetriBatched instead
    JSCUDA_DLL_API
        ErrorType matrixInverseBatched(const cublasHandle_t handle, const MatrixBatch& matAd, MatrixBatch& matAinvd, int* infoArrayd)
    {
        if (matAd.count != matAinvd.count || matAd.numRow != matAd.numCol || matAd.numCol != matAinvd.numRow || matAinvd.numRow != matAinvd.numCol)
            return JC_PARAM_ERROR;
        if (matAd.transposed)
        {
            fprintf(stderr, "matrixGetriBatched Not Support Transposed Matrices Yet.");
            return JC_PARAM_ERROR;
        }

        int n = matAd.numRow;
        int lda = matAd.numRow;
        float ** Aarray = matAd.elementsArray;
        float ** invAarray = matAinvd.elementsArray;
        int lda_inv = matAinvd.numRow;
        int batchSize = matAd.count;

        checkCudaErrors(cublasSmatinvBatched(handle
            , n
            , (const float **)Aarray
            , lda
            , invAarray
            , lda_inv
            , infoArrayd
            , batchSize
            ));

        matAinvd.transposed = false;
        return JC_SUCCESS;
    }


    // TODO: Not tested
    JSCUDA_DLL_API
        ErrorType matrixGeqrfBatched(const cublasHandle_t handle, MatrixBatch& matAd, MatrixBatch& matTaud, int* infoArrayh)
    {
        if (matAd.count != matTaud.count || matAd.numCol != matTaud.numRow || matTaud.numRow != matTaud.numCol)
            return JC_PARAM_ERROR;
        if (matAd.transposed)
        {
            fprintf(stderr, "matrixGetriBatched Not Support Transposed Matrices Yet.");
            return JC_PARAM_ERROR;
        }
        int m = matAd.numRow;
        int n = matAd.numCol;
        float ** Aarray = matAd.elementsArray;
        float ** TauArray = matTaud.elementsArray;
        int lda = matAd.numRow;
        int batchSize = matAd.count;

        checkCudaErrors(cublasSgeqrfBatched(handle
            , m
            , n
            , Aarray
            , lda
            , TauArray
            , infoArrayh
            , batchSize
            ));

        matTaud.transposed = false;
        return JC_SUCCESS;
    }

    // TODO: Not tested
    JSCUDA_DLL_API
        ErrorType matrixGelsBatched(const cublasHandle_t handle, MatrixBatch& matAd, MatrixBatch& matBd, int* infoArrayh)
    {
        if (matAd.count != matBd.count || matAd.numRow != matAd.numCol || matAd.numCol != matBd.numRow || matBd.numRow != matBd.numCol)
            return JC_PARAM_ERROR;
        if (matAd.transposed)
        {
            fprintf(stderr, "matrixGetriBatched Not Support Transposed Matrices Yet.");
            return JC_PARAM_ERROR;
        }
        int m = matAd.numRow;
        int n = matAd.numCol;
        float ** Aarray = matAd.elementsArray;
        float ** Barray = matBd.elementsArray;
        int nrhs = matBd.numCol;
        int lda = matAd.numRow;
        int ldb = matBd.numRow;
        int batchSize = matAd.count;

        cublasOperation_t trans = CUBLAS_OP_N; // TODO: Deal with tranposed matrix

        cublasSgelsBatched(handle
            , trans
            , m
            , n
            , nrhs
            , Aarray
            , lda
            , Barray
            , ldb
            , infoArrayh
            , nullptr
            , batchSize
            );

        matBd.transposed = false;
        return JC_SUCCESS;
    }


}