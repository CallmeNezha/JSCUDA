#include <node.h>
#include "jc_api.h"
#include "jc_parameter.h"
#include "DeviceFloat32Array.h"

#include "matrix_func.h"
#include "vector_func.h"

// Global 
bool                       jcg_bCudaInitialized = false;
jc_cuda::cublasHandle_t    jcg_cublasHandle = nullptr;

namespace jc {


void cudaDeviceInit(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();
    bool ret = jc_cuda::cudaDeviceInit(0, nullptr);
    jcg_bCudaInitialized = ret;
    if (ret) jc_cuda::cublasCreate_t(&jcg_cublasHandle);
    args.GetReturnValue().Set(Boolean::New(isolate,ret));
}

void cudaDeviceReset(const FunctionCallbackInfo<Value>& args) {
    Isolate* isolate = args.GetIsolate();
    jc_cuda::cudaDeviceReset_t();
    if (jcg_bCudaInitialized) jc_cuda::cublasDestroy_t(jcg_cublasHandle);
    jcg_cublasHandle = nullptr;
    jcg_bCudaInitialized = false;
    //args.GetReturnValue().Set(String::NewFromUtf8(isolate, "cuda device reseted.\n"));
}




void Init(Local<Object> exports) {


    NODE_SET_METHOD(exports, "vectorAdd"             , vectorAdd             );
    NODE_SET_METHOD(exports, "vectorCopy"            , vectorCopy            );
    NODE_SET_METHOD(exports, "vectorMulScalar"       , vectorMulScalar       );
    NODE_SET_METHOD(exports, "vectorDot"             , vectorDot             );
    NODE_SET_METHOD(exports, "vectorNorm"            , vectorNorm            );
    NODE_SET_METHOD(exports, "vectorRank"            , vectorRank            );
    NODE_SET_METHOD(exports, "matrixMulScalar"       , matrixMulScalar       );
    NODE_SET_METHOD(exports, "matrixMulVector"       , matrixMulVector       );
    NODE_SET_METHOD(exports, "matrixMulMatrix"       , matrixMulMatrix       );
    NODE_SET_METHOD(exports, "matrixMulMatrixBatched", matrixMulMatrixBatched);

    NODE_SET_METHOD(exports, "cudaDeviceInit" , cudaDeviceInit );
    NODE_SET_METHOD(exports, "cudaDeviceReset", cudaDeviceReset);
    DeviceFloat32Array::Init(exports);
}

NODE_MODULE(jc, Init)

}  // !jc