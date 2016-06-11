#include <node.h>
#include "cuda_common.h"
#include "matrix.h"
#include "vector.h"
namespace jc {



void cudaDeviceInit(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();
    jc_cuda::cudaDeviceInit(0, NULL);
    args.GetReturnValue().Set(String::NewFromUtf8(isolate, "cuda device initialized.\n"));
}

void cudaDeviceReset(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();
    jc_cuda::cudaDeviceReset_t();
    args.GetReturnValue().Set(String::NewFromUtf8(isolate, "cuda device reseted.\n"));
}




void Init(Local<Object> exports) {
    NODE_SET_METHOD(exports, "vectorAdd", vectorAdd);
    NODE_SET_METHOD(exports, "matrixMulMatrix", matrixMulMatrix);
    NODE_SET_METHOD(exports, "cudaDeviceInit", cudaDeviceInit);
    NODE_SET_METHOD(exports, "cudaDeviceReset", cudaDeviceReset);
}

NODE_MODULE(jc, Init)

}  // jc