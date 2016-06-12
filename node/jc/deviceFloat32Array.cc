#include "deviceFloat32Array.h"
#include "cuda_common.h"
#include "jc_parameter.h"
#include "error_message.h"

Persistent<Function> DeviceFloat32Array::constructor;

DeviceFloat32Array::DeviceFloat32Array(uint32 length) 
:m_length(length)
{
    if (m_length != 0)
    {
#ifdef DEBUG_OUTPUT_ON
        printf("Allocating device memory, size%d ...\n", m_length);
#endif
        jc_cuda::cudaMalloc_t((void**)&m_ptrd, m_length*sizeof(float32));
    }
}

DeviceFloat32Array::DeviceFloat32Array(const Local<Float32Array>& f32a)
:m_length(f32a->Length())
{
    if (m_length != 0)
    {
#ifdef DEBUG_OUTPUT_ON
        printf("Allocating device memory and initialized with native Float32Array, size%d ...\n", m_length);
#endif
        jc_cuda::cudaMalloc_t((void**)&m_ptrd, f32a->Buffer()->ByteLength());
        jc_cuda::cudaMemcpyHostToDevice_t(f32a->Buffer()->GetContents().Data()
            , m_ptrd
            , 0
            , f32a->Buffer()->ByteLength()
            );
    }
}

DeviceFloat32Array::~DeviceFloat32Array()
{
    if (m_length != 0)
    {
#ifdef DEBUG_OUTPUT_ON
        printf("Deallocating device memory...\n");
#endif
        jc_cuda::cudaFree_t(m_ptrd);
    }
}

// Embed step
void DeviceFloat32Array::Init(Local<Object> exports)
{
    Isolate* isolate = exports->GetIsolate();

    // Prepare constructor template
    Local<FunctionTemplate> tpl = FunctionTemplate::New(isolate, New);
    tpl->SetClassName(String::NewFromUtf8(isolate, "DeviceFloat32Array"));
    tpl->InstanceTemplate()->SetInternalFieldCount(2); // Tell v8 remains two slots for tag along C++ objects

    // Set Prototype functions in C++
    NODE_SET_PROTOTYPE_METHOD(tpl, "length", length);
    NODE_SET_PROTOTYPE_METHOD(tpl, "createFrom", createFrom);
    constructor.Reset(isolate, tpl->GetFunction());
    exports->Set(String::NewFromUtf8(isolate, "DeviceFloat32Array"), tpl->GetFunction());
}

// Register 'new' behavior in JavaScript
void DeviceFloat32Array::New(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();

    if (args.IsConstructCall())
    {
        if (!jc_bCudaInitialized)
        {
            isolate->ThrowException(Exception::Error(String::NewFromUtf8(isolate, jc::deviceError)));
            return;
        }
        // Invoke as constructor: 'new DeviceFloat32Array(uint32 length)'
        if (args[0]->IsUint32())
        {
            size_t length = (size_t)(args[0]->NumberValue());

            DeviceFloat32Array* obj = new DeviceFloat32Array(length);
            obj->Wrap(args.This());
            args.GetReturnValue().Set(args.This());
            return;
        }
        else if (args[0]->IsFloat32Array())
        {
            Local<Float32Array> f32ah = Local<Float32Array>::Cast(args[0]);

            DeviceFloat32Array* obj = new DeviceFloat32Array(f32ah);
            obj->Wrap(args.This());
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

// Prototype function
// return length
void DeviceFloat32Array::length(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();
    DeviceFloat32Array* obj = ObjectWrap::Unwrap<DeviceFloat32Array>(args.Holder());
    args.GetReturnValue().Set(Number::New(isolate, obj->getLength()));
}

// Prototype function
// Create and initialize buffer value by host Float32Array
void DeviceFloat32Array::createFrom(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();
    if (!args[0]->IsFloat32Array())
    {
        isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, jc::typArgError)));
        return;
    }
    const uint32 argc = 1;
    Local<Value> argv[argc] = { args[0] }; // Arguments array explicitly
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Context> context = isolate->GetCurrentContext();
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked();

    args.GetReturnValue().Set(instance);

}
