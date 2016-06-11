#include "cuda_common.h"
#include <cstdio>
#include <vector>
#include <memory>
#include <windows.h>

#define dumpMatrix_m(mat)         printf("Dumping matrix %s: \n",#mat);dumpMatrix(mat);

using namespace jc_cuda;
void dumpMatrix(Matrix&);

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


void matrixMultiplyTest();
int main(int argc, char **argv)
{
    matrixMultiplyTest();
}
void matrixMultiplyTest()
{

    Matrix matAh = Matrix{ 2, 3, new float[6] };
    Matrix matBh = Matrix{ 3, 5, new float[15] };
    Matrix matCh = Matrix{ 2, 5, new float[10] };
    for (unsigned int i = 0; i < 6; ++i)
    {
        matAh.elements[i] = (float)i;
    }
    for (unsigned int i = 0; i < 15; ++i)
    {
        matBh.elements[i] = (float)i;
    }
    dumpMatrix_m(matAh);
    dumpMatrix_m(matBh);
    Matrix matAd = matAh;
    matAd.elements = nullptr;
    Matrix matBd = matBh;
    matBd.elements = nullptr;
    Matrix matCd = matCh;
    matCd.elements = nullptr;
    cudaMalloc_t((void**)&matAd.elements, 6 * sizeof(float));
    cudaMalloc_t((void**)&matBd.elements, 15 * sizeof(float));
    cudaMalloc_t((void**)&matCd.elements, 10 * sizeof(float));
    cudaMemcpyHostToDevice_t(matAh.elements, matAd.elements, 0, 6 * sizeof(float));
    cudaMemcpyHostToDevice_t(matBh.elements, matBd.elements, 0, 15 * sizeof(float));

    cublasHandle_t handle = nullptr;
#ifdef _DEBUG
    printf("Entering cuBLAS context...\n");
#endif
    cublasCreate_t(&handle);
#ifdef _DEBUG
    printf("Matrix multiplying...\n");
#endif
    matrixMulMatrix_blas(handle, matAd, matBd, matCd);
    matrixMulMatrix_blas(handle, matAd, matBd, matCd);
    cudaMemcpyDeviceToHost_t(matCd.elements, matCh.elements, 0, 10 * sizeof(float));
    dumpMatrix_m(matCh);
#ifdef _DEBUG
    printf("Leaving cuBLAS context...\n");
#endif
    cublasDestroy_t(handle);
    cudaFree_t(matAd.elements);
    cudaFree_t(matBd.elements);
    cudaFree_t(matCd.elements);
    delete matAh.elements;
    delete matBh.elements;
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


bool testVectorAdd(int argc, char **argv)
{
    cudaDeviceInit(argc, argv);
    auto v1 = std::unique_ptr<std::vector<float>>(new std::vector<float>(10, 3));
    auto v2 = std::unique_ptr<std::vector<float>>(new std::vector<float>(10, 4));
    auto v3 = std::unique_ptr<std::vector<float>>(new std::vector<float>(10, 6));
    for (unsigned int i = 0; i < 5; i++)
    {
        printf("%f ,", v3->at(i));
    }
    StartCounter();
    float *v1d, *v2d, *v3d;
    unsigned int size = 10 * sizeof(float);
    cudaMalloc_t((void**)&v1d, size);
    cudaMalloc_t((void**)&v2d, size);
    cudaMalloc_t((void**)&v3d, size);
    cudaMemcpyHostToDevice_t(&(v1->at(0)), v1d, 0, size);
    cudaMemcpyHostToDevice_t(&(v2->at(0)), v2d, 0, size);
    vectorAdd(v1d, v2d, v3d, 10);
    cudaMemcpyDeviceToHost_t(v3d, &(v3->at(0)), 0, size);
    double consumeTime = GetCounter();
    for (unsigned int i = 0; i < 5; i++)
    {
        printf("%f ,", v3->at(i));
    }
    printf("\n consumeTime: %f", consumeTime);

    getchar();

    cudaFree_t(v1d);
    cudaFree_t(v2d);
    cudaFree_t(v3d);
    cudaDeviceReset_t();
    return true;
}