#include "jc_api.h"
#include <cstdio>
#include <vector>
#include <memory>
#include <windows.h>
using namespace jc_cuda;

#define dumpMatrix_m(mat)         printf("Dumping matrix %s: \n",#mat);dumpMatrix(mat);

void dumpMatrix(Matrix&);

#define dumpHostVector_m(ptr,count)   printf("Dumping vector %s: \n",#ptr);dumpHostVector(ptr,count);
void dumpHostVector(const float *ptr, unsigned int count)
{
    printf("[ ");
    for (unsigned int i = 0; i < count; ++i)
    {
        printf("%f ", ptr[i]);
    }
    printf("]\n");
}

double PCFreq = 0.0;
__int64 CounterStart = 0;
void StartCounter()
{
    LARGE_INTEGER li;
    if (!QueryPerformanceFrequency(&li))
        fprintf(stderr, "QueryPerformanceFrequency failed!\n");

    PCFreq = double(li.QuadPart) / 1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}
double GetCounter()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart - CounterStart) / PCFreq;
}


bool testBatchMatrixMultiply(int argc, char **argv);
int main(int argc, char **argv)
{
    testBatchMatrixMultiply(argc, argv);
    getchar();
    return 0;
}


void dumpMatrix(Matrix& mat)
{
    printf("numRow: %d, numCol: %d\n", mat.numRow, mat.numCol);
    if (0 == mat.numRow || 0 == mat.numCol) return;
    for (unsigned int r = 0; r < mat.numRow; ++r)
    {
        printf("%d:[ ",r);
        for (unsigned int c = 0; c < mat.numCol; ++c)
        {
            printf("%.2f", mat.elements[c*mat.numRow + r]);
            if (c != mat.numCol - 1) printf(",");
        }
        printf(" ]\n");
    }
    printf("\n");
}

bool testBatchMatrixMultiply(int argc, char **argv)
{
    cudaDeviceInit(0, nullptr);
    cublasHandle_t handle;
    cublasCreate_t(&handle);

    auto vAh = new float[6];
    auto vBh = new float[6];
    auto vCh = new float[4];

    for (int i = 0; i < 6; i++)
    {
        vAh[i] = i;
        vBh[i] = i;
    }
    float *vAd, *vBd, *vCd;
    float **vAdb, **vBdb, **vCdb;


    size_t memsize = 6 * sizeof(float);
    cudaMalloc_t((void**)&vAd, memsize);
    cudaMalloc_t((void**)&vBd, memsize);
    cudaMalloc_t((void**)&vCd, memsize);
    cudaMemcpyHostToDevice_t(vAh, vAd, 0, 0, memsize);
    cudaMemcpyHostToDevice_t(vBh, vBd, 0, 0, memsize);

    float** a = (float**)malloc(sizeof(float*));
    float** b = (float**)malloc(sizeof(float*));
    float** c = (float**)malloc(sizeof(float*));

    a[0] = vAd;
    b[0] = vBd;
    c[0] = vCd;

    cudaMalloc_t((void**)&vAdb, sizeof(float*));
    cudaMalloc_t((void**)&vBdb, sizeof(float*));
    cudaMalloc_t((void**)&vCdb, sizeof(float*));

    cudaMemcpyHostToDevice_t(a, vAdb, 0, 0, sizeof(float*));
    cudaMemcpyHostToDevice_t(b, vBdb, 0, 0, sizeof(float*));
    cudaMemcpyHostToDevice_t(c, vCdb, 0, 0, sizeof(float*));

    MatrixBatch mbA{ 3, 2, vAdb, true, 1 };
    MatrixBatch mbB{ 2, 3, vBdb, true, 1 };
    MatrixBatch mbC{ 2, 2, vCdb, false, 1 };

    matrixMulMatrixBatched(handle, mbA, mbB, mbC);

    cudaMemcpyDeviceToHost_t(vCd, vCh, 0, 0, 4 * sizeof(float));
    dumpHostVector_m(vCh, 4);


    cudaFree_t(vAdb);
    cudaFree_t(vBdb);
    cudaFree_t(vCdb);
    cudaFree_t(vAd);
    cudaFree_t(vBd);
    cudaFree_t(vCd);


    delete[] vAh;
    delete[] vBh;
    delete[] vCh;
    free(a);
    free(b);
    free(c);
    cudaDeviceReset_t();
    return true;
}


bool testVectorAdd(int argc, char **argv)
{
    cudaDeviceInit(0, nullptr);
    cublasHandle_t handle;
    cublasCreate_t(&handle);
    auto vAh = new float[10];
    auto vBh = new float[10];
    for (int i = 0; i < 10; i++)
    {
        vAh[i] = i;
        vBh[i] = i;
    }
    float *vAd, *vBd;
    cudaMalloc_t((void**)&vAd, 10 * sizeof(float));
    cudaMalloc_t((void**)&vBd, 10 * sizeof(float));

    cudaMemcpyHostToDevice_t(vAh, vAd, 0, 0, 10 * sizeof(float));
    cudaMemcpyHostToDevice_t(vBh, vBd, 0, 0, 10 * sizeof(float));

    Vector v1{ 10, vAd };
    Vector v2{ 10, vBd };

    vectorAdd(handle, v2, v1);
    cudaMemcpyDeviceToHost_t(vAd, vAh, 0, 0, 10 * sizeof(float));
    cudaMemcpyDeviceToHost_t(vBd, vBh, 0, 0, 10 * sizeof(float));
    cublasDestroy_t(handle);
    cudaDeviceReset_t();
    dumpHostVector_m(vAh, 10);
    dumpHostVector_m(vBh, 10);

    delete[] vAh;
    delete[] vBh;
    cudaDeviceReset_t();
    return true;

}