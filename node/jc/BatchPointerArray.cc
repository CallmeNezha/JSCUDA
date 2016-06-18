#include "BatchPointerArray.h"
#include "jc_api.h"
#include "jc_parameter.h"
#include "error_message.h"
#include "DeviceFloat32Array.h"
#include <memory>

#define STRINGFY(val)                #val

Persistent<Function> BatchPointerArray::constructor;

BatchPointerArray::BatchPointerArray(size_t size, PointerType type, void *ptrd)
: m_size(size)
, m_type(type)
, m_ptrd(ptrd)
{

}

BatchPointerArray::~BatchPointerArray()
{
    if (0 != m_size && nullptr != m_ptrd)
    {
        printf("Destruct BatchPointerArray contains %d float32 * array \n"
            , getLength()
            );
        jc_cuda::cudaFree_t(m_ptrd);
    }
}

void BatchPointerArray::New(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    if (args.IsConstructCall())
    {
        if (!jcg_bCudaInitialized)
        {
            isolate->ThrowException(Exception::Error(String::NewFromUtf8(isolate, jc::deviceError)));
            return;
        }
        // Invoke as constructor: 'new DeviceFloat32Array(uint32 length)'
        if (args[0]->IsArray())
        {
            auto f32aa = Local<Array>::Cast(args[0]);
            void* hostBuffer = malloc(f32aa->Length() * sizeof(float32*));
            for (uint32 i = 0; i < f32aa->Length(); ++i)
            {
                DeviceFloat32Array* ptr = ObjectWrap::Unwrap<DeviceFloat32Array>(f32aa->Get(i)->ToObject());
                if (nullptr == ptr)
                {
                    free(hostBuffer);
                    isolate->ThrowException(Exception::Error(String::NewFromUtf8(isolate, jc::typArgError)));
                    return;
                }
                ((float32**)hostBuffer)[i] = ptr->getData();
            }
            void* deviceBuffer;
            jc_cuda::cudaMalloc_t(&deviceBuffer, f32aa->Length() * sizeof(float32*));
            jc_cuda::cudaMemcpyHostToDevice_t(hostBuffer, deviceBuffer, 0, 0, f32aa->Length() * sizeof(float32*));
            free(hostBuffer);

            BatchPointerArray* obj = new BatchPointerArray(f32aa->Length() * sizeof(float32*), Float32_Pointer, deviceBuffer);
            
            obj->Wrap(args.This());
            // Add properties here
            args.This()->Set(String::NewFromUtf8(isolate, "length"), Uint32::NewFromUnsigned(isolate, (uint32)obj->getLength()));
            args.This()->Set(String::NewFromUtf8(isolate, "typeSize"), Uint32::NewFromUnsigned(isolate, (uint32)obj->getTypeSize()));
            args.This()->Set(String::NewFromUtf8(isolate, "type"), String::NewFromUtf8(isolate, STRINGFY(Float32_Pointer)));
            args.GetReturnValue().Set(args.This());
            return;
        }
        else
        {
            isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, jc::typArgError)));
            return;
        }
    }
    else
    {
        isolate->ThrowException(Exception::SyntaxError(String::NewFromUtf8(isolate, jc::noNewError)));
        return;
    }
}

void BatchPointerArray::destroy(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();
    BatchPointerArray* obj = ObjectWrap::Unwrap<BatchPointerArray>(args.Holder());
    jc_cuda::cudaFree_t(obj->m_ptrd);
    obj->m_ptrd = nullptr;
    obj->m_size = 0;
    args.GetReturnValue().Set(Undefined(isolate));
}

void BatchPointerArray::Init(Local<Object> exports)
{
    Isolate* isolate = exports->GetIsolate();

    // Prepare constructor template
    Local<FunctionTemplate> tpl = FunctionTemplate::New(isolate, New);
    tpl->SetClassName(String::NewFromUtf8(isolate, "BatchPointerArray"));
    tpl->InstanceTemplate()->SetInternalFieldCount(1); // Tell v8 remains one slots for C++ pointer

    // Set Prototype functions in C++
    //NODE_SET_PROTOTYPE_METHOD(tpl, "length"  , length  );

    NODE_SET_PROTOTYPE_METHOD(tpl, "destroy", destroy);

    constructor.Reset(isolate, tpl->GetFunction());

    exports->Set(String::NewFromUtf8(isolate, "BatchPointerArray"), tpl->GetFunction());
}

size_t BatchPointerArray::getTypeSize() const
{
    switch (m_type)
    {
    case BatchPointerArray::Float32_Pointer:
        return sizeof(float32*);
    default:
        return 0;
    }
}

