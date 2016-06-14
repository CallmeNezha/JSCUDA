#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <node.h>
#include "jc_api.h"
#include "jc_helper.h"
#include "jc_parameter.h"
#include "error_message.h"

using namespace v8;

// All of the verify are done in Javascript-part to ease successive maintenance
// meanwhile against the craze changes of v8, node API along with every version

#define dumpHostVector_m(ptr,count)   printf("Dumping vector %s: \n",#ptr);dumpHostVector(ptr,count);
void dumpHostVector(float32 *ptr, uint32 count)
{
    printf("[ ");
    for (unsigned int i = 0; i < count; ++i)
    {
        printf("%f ", ptr[i]);
    }
    printf("]\n");
}
                  
// vAd += vBd
void vectorAdd(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    if (nullptr == jcg_cublasHandle)   
    {                                                           
        isolate->ThrowException(Exception::ReferenceError(String::NewFromUtf8(isolate, jc::cublasError)));    
        return;                                                 
    }

    jc_cuda::Vector vAd = unwrapVector(isolate, args[0]);
    jc_cuda::Vector vBd = unwrapVector(isolate, args[1]);

    if (vAd.length != vBd.length || vAd.elements == nullptr || vBd.elements == nullptr)
    {
        isolate->ThrowException(Exception::ReferenceError(String::NewFromUtf8(isolate, jc::cublasError)));
        return;
    }
#if 0
    float32 *vAh, *vBh;
    vAh = new float32[vAd.length];
    vBh = new float32[vBd.length];
    jc_cuda::cudaMemcpyDeviceToHost_t(vAd.elements, vAh, 0, 0, vAd.length*sizeof(float32));
    jc_cuda::cudaMemcpyDeviceToHost_t(vBd.elements, vBh, 0, 0, vBd.length*sizeof(float32));
    dumpHostVector_m(vAh, 10);
    dumpHostVector_m(vBh, 10);
    jc_cuda::cublasHandle_t handle = nullptr;
    jc_cuda::cublasCreate_t(&handle);
#endif

    checkJCErrors(jc_cuda::vectorAdd(jcg_cublasHandle, vAd, vBd));
#if 0
    jc_cuda::cudaMemcpyDeviceToHost_t(vAd.elements, vAh, 0, 0, vAd.length*sizeof(float32));
    jc_cuda::cudaMemcpyDeviceToHost_t(vBd.elements, vBh, 0, 0, vBd.length*sizeof(float32));
    dumpHostVector_m(vAh, 10);
    dumpHostVector_m(vBh, 10);
    jc_cuda::cublasDestroy_t(handle);
#endif

}

// vAd = vBd
void vectorCopy(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    CUBLAS_HANDLE_CHECK_RETURN;

    jc_cuda::Vector vAd = unwrapVector(isolate, args[0]);
    jc_cuda::Vector vBd = unwrapVector(isolate, args[1]);

    checkJCErrors(jc_cuda::vectorCopy(jcg_cublasHandle, vAd, vBd));
}

// swap vAd <==> vBd
void vectorSwap(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    CUBLAS_HANDLE_CHECK_RETURN;

    jc_cuda::Vector vAd = unwrapVector(isolate, args[0]);
    jc_cuda::Vector vBd = unwrapVector(isolate, args[1]);

    checkJCErrors(jc_cuda::vectorSwap(jcg_cublasHandle, vAd, vBd));
}

// vAd dot vBd
void vectorDot(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    CUBLAS_HANDLE_CHECK_RETURN;

    jc_cuda::Vector vAd = unwrapVector(isolate, args[0]);
    jc_cuda::Vector vBd = unwrapVector(isolate, args[1]);

    float32 result = 0.f;
    checkJCErrors(jc_cuda::vectorDot(jcg_cublasHandle, vAd, vBd, &result));
    args.GetReturnValue().Set(Number::New(isolate, result));
}

// vd Euclidean norm
void vectorNorm(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    CUBLAS_HANDLE_CHECK_RETURN;

    jc_cuda::Vector vd = unwrapVector(isolate, args[0]);

    float32 result = 0.f;
    checkJCErrors(jc_cuda::vectorNorm(jcg_cublasHandle, vd, &result));
    args.GetReturnValue().Set(Number::New(isolate, result));
}

// vd = scalar * vd
void vectorMulScalar(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    CUBLAS_HANDLE_CHECK_RETURN;

    jc_cuda::Vector vd = unwrapVector(isolate, args[0]);
    float32 scalar = (float32)args[1]->NumberValue();
    checkJCErrors(jc_cuda::vectorMulScalar(jcg_cublasHandle, vd, &scalar));
}

// matd = vAd * vBd^T
void vectorRank(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    CUBLAS_HANDLE_CHECK_RETURN;

    jc_cuda::Vector vAd = unwrapVector(isolate, args[0]);
    jc_cuda::Vector vBd = unwrapVector(isolate, args[1]);
    jc_cuda::Matrix matd = unwrapMatrix(isolate, args[2]);
    checkJCErrors(jc_cuda::vectorRank(jcg_cublasHandle, vAd, vBd, matd));
}



#endif //!__VECTOR_H__
