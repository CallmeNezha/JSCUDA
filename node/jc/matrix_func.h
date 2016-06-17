#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <node.h>
#include "jc_api.h"
#include "jc_helper.h"
#include "jc_parameter.h"
#include "error_message.h"

using namespace v8;


// matd = scalar * matd
void matrixMulScalar(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    CUBLAS_HANDLE_CHECK_RETURN;

    jc_cuda::Matrix matd = unwrapMatrix(isolate, args[0]);
    float32 scalar = (float32)args[1]->NumberValue();
    checkJCErrors(jc_cuda::matrixMulScalar(jcg_cublasHandle
        , matd
        , &scalar
        ));
}

// vBd = matd(^T) * vAd
void matrixMulVector(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    CUBLAS_HANDLE_CHECK_RETURN;

    jc_cuda::Matrix matd = unwrapMatrix(isolate, args[0]);
    jc_cuda::Vector vAd = unwrapVector(isolate, args[1]);
    jc_cuda::Vector vBd = unwrapVector(isolate, args[2]);
    checkJCErrors(jc_cuda::matrixMulVector(jcg_cublasHandle
        , matd
        , vAd
        , vBd
        ));

}

// matCd = matAd(^T) * matBd(^T)
void matrixMulMatrix(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    CUBLAS_HANDLE_CHECK_RETURN;

    jc_cuda::Matrix matAd = unwrapMatrix(isolate, args[0]);
    jc_cuda::Matrix matBd = unwrapMatrix(isolate, args[1]);
    jc_cuda::Matrix matCd = unwrapMatrix(isolate, args[2]);
    checkJCErrors(jc_cuda::matrixMulMatrix(jcg_cublasHandle
        , matAd
        , matBd
        , matCd
        ));
}

// matCd = matAd(^T) * matBd(^T) batched
// TODO: complete batch operation
void matrixMulMatrixBatched(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    CUBLAS_HANDLE_CHECK_RETURN;

    auto matAbatch = Local<Object>::Cast(args[0]);
    auto matBbatch = Local<Object>::Cast(args[1]);
    auto matCbatch = Local<Object>::Cast(args[2]);

    auto numRow = String::NewFromUtf8(isolate, "numRow");
    auto numCol = String::NewFromUtf8(isolate, "numCol");
    auto transposed = String::NewFromUtf8(isolate, "transposed");
    auto count = String::NewFromUtf8(isolate, "count");
    auto elementsArray = String::NewFromUtf8(isolate, "elementsArray");

    auto matAea = Local<Array>::Cast(matAbatch->Get(elementsArray));

    for (uint32 i = 0; i < matAea->Length(); i++)
    {
        DeviceFloat32Array* vecf32ad = node::ObjectWrap::Unwrap<DeviceFloat32Array>(matAea->Get(i)->ToObject());
        vecf32ad->getData();
    }
    
    jc_cuda::MatrixBatch mb{};

    //checkJCErrors(jc_cuda::matrixMulMatrixBatched(jcg_cublasHandle
    //    , matAd
    //    , transA
    //    , matBd
    //    , transB
    //    , matCd
    //    ));
}
#endif //!__MATRIX_H__