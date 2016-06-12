#include <node.h>
#include "cuda_common.h"
#include "matrix.h"
#include "vector.h"
#include "deviceFloat32Array.h"
#include "jc_parameter.h"

// Global 
bool jc_bCudaInitialized = false;

namespace jc {


void cudaDeviceInit(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();
    bool ret = jc_cuda::cudaDeviceInit(0, NULL);
    jc_bCudaInitialized = ret;
    args.GetReturnValue().Set(Boolean::New(isolate,ret));
}

void cudaDeviceReset(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();
    jc_cuda::cudaDeviceReset_t();
    jc_bCudaInitialized = false;
    //args.GetReturnValue().Set(String::NewFromUtf8(isolate, "cuda device reseted.\n"));
}




void Init(Local<Object> exports) {
    NODE_SET_METHOD(exports, "vectorAdd", vectorAdd);
    NODE_SET_METHOD(exports, "matrixMulMatrix", matrixMulMatrix);
    NODE_SET_METHOD(exports, "cudaDeviceInit", cudaDeviceInit);
    NODE_SET_METHOD(exports, "cudaDeviceReset", cudaDeviceReset);
    DeviceFloat32Array::Init(exports);
}

NODE_MODULE(jc, Init)

}  // !jc