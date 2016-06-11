#ifndef __CUDA_COMMON_KERNEL_CUH__
#define __CUDA_COMMON_KERNEL_CUH__

#ifndef UINT_T
typedef unsigned int        uint;
#endif // !UINT_T


#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <device_launch_parameters.h>

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

//Default parameter, may scalar by 2 for Fermi or above
unsigned int  U_NUM_THREAD = 256;
unsigned int  U_NUM_BLOCKSIZE = 16;


__global__ void vectorAddD(const float* v1, const float* v2, float* vout, uint size)
{
    uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index > size) return;
    vout[index] = v1[index] + v2[index];
}

__device__ Matrix getSubMatrix(Matrix mat, uint row, uint col, uint blockSize)
{
    Matrix rMat;
    rMat.numRow = blockSize;
    rMat.numCol = blockSize;
//    rMat.stride = mat.stride;
//    rMat.elements = &mat.elements[row * blockSize + rMat.stride * blockSize * col];
    return rMat;
}

__host__ __device__ inline uint ceilfuint(float f)
{
    return (uint)ceilf(f);
}





#endif //!__CUDA_COMMON_KERNEL_CUH__
