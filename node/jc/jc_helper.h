#ifndef __JC_HELPER_H__
#define __JC_HELPER_H__
#include <node.h>
#include "DeviceFloat32Array.h"
#include "BatchPointerArray.h"

using namespace v8;

#define CUBLAS_HANDLE_CHECK_RETURN     if (!jcg_cublasHandle)   \
{                                                           \
    isolate->ThrowException(Exception::ReferenceError(      \
        String::NewFromUtf8(isolate, jc::cublasError)));    \
    return;                                                 \
}



template< typename T >
void checkReturnValue(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "JC error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), "ErrorCode refer jc_api.h", func);
    }
}

#define checkJCErrors(val)           checkReturnValue ( (val), #val, __FILE__, __LINE__ )


jc_cuda::Vector unwrapVector(Isolate* isolate, v8::Local<Value>& arg)
{
    auto vec = Local<Object>::Cast(arg);
    auto length = String::NewFromUtf8(isolate, "length");
    auto elements = String::NewFromUtf8(isolate, "elements");
    DeviceFloat32Array* vecf32ad = node::ObjectWrap::Unwrap<DeviceFloat32Array>(vec->Get(elements)->ToObject());
    jc_cuda::Vector vecd{ vec->Get(length)->Uint32Value(), vecf32ad->getData() };
    return vecd;
}

jc_cuda::Matrix unwrapMatrix(Isolate* isolate, v8::Local<Value>& arg)
{
    auto mat = Local<Object>::Cast(arg);
    auto numRow = String::NewFromUtf8(isolate, "numRow");
    auto numCol = String::NewFromUtf8(isolate, "numCol");
    auto elements = String::NewFromUtf8(isolate, "elements");
    auto transposed = String::NewFromUtf8(isolate, "transposed");
    DeviceFloat32Array* matf32ad = node::ObjectWrap::Unwrap<DeviceFloat32Array>(mat->Get(elements)->ToObject());
    
    uint32 numRow_r, numCol_r; // Restore original number of row, column
    bool transposed_t = mat->Get(transposed)->BooleanValue();
    if (transposed_t)
        numRow_r = mat->Get(numCol)->Uint32Value(), numCol_r = mat->Get(numRow)->Uint32Value();
    else
        numRow_r = mat->Get(numRow)->Uint32Value(), numCol_r = mat->Get(numCol)->Uint32Value();

    jc_cuda::Matrix matd{ numRow_r
        , numCol_r
        , matf32ad->getData()
        , transposed_t
        };
    return matd;
}

jc_cuda::MatrixBatch unwrapMatrixBatch(Isolate* isolate, v8::Local<Value>& arg)
{
    auto matBatch = Local<Object>::Cast(arg);
    auto numRow = String::NewFromUtf8(isolate, "numRow");
    auto numCol = String::NewFromUtf8(isolate, "numCol");
    auto transposed = String::NewFromUtf8(isolate, "transposed");
    auto count = String::NewFromUtf8(isolate, "count");
    auto batchPointerArray = String::NewFromUtf8(isolate, "batchPointerArray");
    BatchPointerArray* bpad = node::ObjectWrap::Unwrap<BatchPointerArray>(matBatch->Get(batchPointerArray)->ToObject());

    uint32 numRow_r, numCol_r; // Restore original number of row, column
    bool transposed_t = matBatch->Get(transposed)->BooleanValue();
    if (transposed_t)
        numRow_r = matBatch->Get(numCol)->Uint32Value(), numCol_r = matBatch->Get(numRow)->Uint32Value();
    else
        numRow_r = matBatch->Get(numRow)->Uint32Value(), numCol_r = matBatch->Get(numCol)->Uint32Value();

    jc_cuda::MatrixBatch mbd{ numRow_r
        , numCol_r
        , (float**)bpad->getData()
        , transposed_t
        , matBatch->Get(count)->Uint32Value()
        };
    return mbd;
}

#endif //!__JC_HELPER_H__