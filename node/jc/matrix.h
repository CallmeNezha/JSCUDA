#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <node.h>
#include "cuda_common.h"
using v8::Exception;
using v8::FunctionCallbackInfo;
using v8::Isolate;
using v8::Local;
using v8::Array;
using v8::Float32Array;
using v8::ArrayBuffer;
using v8::Number;
using v8::Object;
using v8::String;
using v8::Value;


void matrixMulMatrix(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();

    // Check the number of arguments passed.
    if (args.Length() != 3) {
        // Throw an Error that is passed back to JavaScript
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Wrong number of arguments")));
        return;
    }

    jc_cuda::Matrix matA, matB, matC;

    // Check the matrix size
    Local<Object> matAh = Local<Object>::Cast(args[0]); // All *T is T's pointer return
    Local<Object> matBh = Local<Object>::Cast(args[1]);
    Local<Object> matCh = Local<Object>::Cast(args[2]);
    bool ret = false;

    auto numRow = String::NewFromUtf8(isolate, "numRow"), numCol = String::NewFromUtf8(isolate, "numCol");
    matA.numRow = (unsigned int)matAh->Get(numRow)->NumberValue();
    matA.numCol = (unsigned int)matAh->Get(numCol)->NumberValue();
    matB.numRow = (unsigned int)matBh->Get(numRow)->NumberValue();
    matB.numCol = (unsigned int)matBh->Get(numCol)->NumberValue();
    matC.numRow = (unsigned int)matCh->Get(numRow)->NumberValue();
    matC.numCol = (unsigned int)matCh->Get(numCol)->NumberValue();

    if (matA.numCol != matB.numRow || matA.numRow != matC.numRow || matB.numCol != matC.numCol)
    {
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Wrong Matrix Size")));
        return;
    }

    // Check elements type & buffer size
    auto elements = String::NewFromUtf8(isolate, "elements");

    // Check the argument types
    if (!matAh->Get(elements)->IsFloat32Array() || !matBh->Get(elements)->IsFloat32Array() || !matCh->Get(elements)->IsFloat32Array())
    {
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Wrong Matrix elements type")));
        return;
    }

    Local<Float32Array> matAeh = Local<Float32Array>::Cast(matAh->Get(elements));
    Local<Float32Array> matBeh = Local<Float32Array>::Cast(matBh->Get(elements));
    Local<Float32Array> matCeh = Local<Float32Array>::Cast(matCh->Get(elements));
    if (matAeh->Length() != matA.numRow * matA.numCol
        || matBeh->Length() != matB.numRow * matB.numCol
        || matCeh->Length() != matC.numRow * matC.numCol)
        ret = true;

    if (ret)
    {
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Wrong arguments")));
        return;
    }


    // Perform the operation
    Local<ArrayBuffer> v1bh = matAeh->Buffer();
    Local<ArrayBuffer> v2bh = matBeh->Buffer();
    Local<ArrayBuffer> v3bh = matCeh->Buffer();
    float *v1d, *v2d, *v3d;
    jc_cuda::cudaMalloc_t((void**)&v1d, sizeof(float) * (matAeh->Length() + matBeh->Length() + matCeh->Length()));
    v2d = &(v1d[matAeh->Length()]);
    v3d = &(v1d[matAeh->Length() + matBeh->Length()]);
    jc_cuda::cudaMemcpyHostToDevice_t(v1bh->GetContents().Data()
        , v1d
        , 0
        , matAeh->Length() * sizeof(float)
        );
    jc_cuda::cudaMemcpyHostToDevice_t(v2bh->GetContents().Data()
        , v2d
        , 0
        , matBeh->Length() * sizeof(float)
        );
    matA.elements = v1d;
    matB.elements = v2d;
    matC.elements = v3d;

    jc_cuda::cublasHandle_t handle = nullptr;
    jc_cuda::cublasCreate_t(&handle);
    jc_cuda::matrixMulMatrix_blas(handle, matA, matB, matC);
    jc_cuda::cudaMemcpyDeviceToHost_t(v3d
        , v3bh->GetContents().Data()
        , 0
        , matCeh->Length() * sizeof(float)
        );
    jc_cuda::cublasDestroy_t(handle);
    jc_cuda::cudaFree_t(v1d);

    //// Set the return value (using the passed in
    //// FunctionCallbackInfo<Value>&)
    //// args.GetReturnValue().Set(num);
}


#endif //!__MATRIX_H__