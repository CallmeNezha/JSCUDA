#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <node.h>
#include "cuda_common.h"
using namespace v8;

// This is the implementation of the "add" method
// Input arguments are passed using the
// const FunctionCallbackInfo<Value>& args struct
void vectorAdd(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();

    // Check the number of arguments passed.
    if (args.Length() != 3) {
        // Throw an Error that is passed back to JavaScript
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Wrong number of arguments")));
        return;
    }

    // Check the argument types
    if (!args[0]->IsFloat32Array() || !args[1]->IsFloat32Array() || !args[2]->IsFloat32Array()) {
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Wrong arguments")));
        return;
    }

    Local<Float32Array> v1h = Local<Float32Array>::Cast(args[0]); // All *T is T's pointer return
    Local<Float32Array> v2h = Local<Float32Array>::Cast(args[1]);
    Local<Float32Array> v3h = Local<Float32Array>::Cast(args[2]);
    if (!(v1h->Length() == v2h->Length() && v2h->Length() == v3h->Length()))
    {
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "Wrong length of arrays")));
        return;
    }

    // Perform the operation
    Local<ArrayBuffer> v1bh = v1h->Buffer();
    Local<ArrayBuffer> v2bh = v2h->Buffer();
    Local<ArrayBuffer> v3bh = v3h->Buffer();
    float *v1d, *v2d, *v3d;
    jc_cuda::cudaMalloc_t((void**)&v1d, (v1h->Length() + v2h->Length() + v3h->Length()) * sizeof(float));
    v2d = &(v1d[v1h->Length()]);
    v3d = &(v1d[v1h->Length() + v2h->Length()]);
    jc_cuda::cudaMemcpyHostToDevice_t(v1bh->GetContents().Data()
        , v1d
        , 0
        , v1h->Length() * sizeof(float)
        );
    jc_cuda::cudaMemcpyHostToDevice_t(v2bh->GetContents().Data()
        , v2d
        , 0
        , v2h->Length() * sizeof(float)
        );
    jc_cuda::vectorAdd(v1d, v2d, v3d, (unsigned int)v1h->Length());
    jc_cuda::cudaMemcpyDeviceToHost_t(v3d
        , v3bh->GetContents().Data()
        , 0
        , v3h->Length() * sizeof(float)
        );
    jc_cuda::cudaFree_t(v1d);

    // Set the return value (using the passed in
    // FunctionCallbackInfo<Value>&)
    // args.GetReturnValue().Set(num);
}



#endif //!__VECTOR_H__
