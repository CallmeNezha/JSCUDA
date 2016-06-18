#include "DeviceFloat32Array.h"
#include "jc_api.h"
#include "jc_parameter.h"
#include "error_message.h"

Persistent<Function> DeviceFloat32Array::constructor;

DeviceFloat32Array::DeviceFloat32Array(size_t length) 
: m_length(length)
{
    if (0 != m_length)
    {
#if DEBUG_OUTPUT_ON
        printf("Allocating device memory, size: %d ...\n", m_length);
#endif
        jc_cuda::cudaMalloc_t((void **)&m_ptrd, m_length * sizeof(float32));
    }
}

DeviceFloat32Array::DeviceFloat32Array(const Local<Float32Array>& f32a)
:m_length(f32a->Length())
{
    if (0 != m_length)
    {
#if DEBUG_OUTPUT_ON
        printf("Allocating device memory and initialized with native Float32Array, size%d ...\n", m_length);
#endif
        jc_cuda::cudaMalloc_t((void**)&m_ptrd, f32a->Buffer()->ByteLength());
        jc_cuda::cudaMemcpyHostToDevice_t(f32a->Buffer()->GetContents().Data()
            , m_ptrd
            , 0
            , 0
            , f32a->Buffer()->ByteLength()
            );
    }
}

DeviceFloat32Array::~DeviceFloat32Array()
{
    if (0 != m_length && nullptr != m_ptrd)
    {
#if DEBUG_OUTPUT_ON
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
    tpl->InstanceTemplate()->SetInternalFieldCount(1); // Tell v8 remains one slots for C++ pointer

    // Set Prototype functions in C++
    //NODE_SET_PROTOTYPE_METHOD(tpl, "length"  , length  );
    
    NODE_SET_PROTOTYPE_METHOD(tpl, "swap", swap);
    NODE_SET_PROTOTYPE_METHOD(tpl, "copy", copy);
    NODE_SET_PROTOTYPE_METHOD(tpl, "copyFrom", copyFrom);
    NODE_SET_PROTOTYPE_METHOD(tpl, "copyTo"  , copyTo  );
    NODE_SET_PROTOTYPE_METHOD(tpl, "destroy",  destroy);

    constructor.Reset(isolate, tpl->GetFunction());

    exports->Set(String::NewFromUtf8(isolate, "DeviceFloat32Array"), tpl->GetFunction());
}

// Register 'new' behavior in JavaScript
void DeviceFloat32Array::New(const FunctionCallbackInfo<Value>& args)
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
        if (args[0]->IsUint32())
        {
            size_t length = (size_t)(args[0]->Uint32Value());

            DeviceFloat32Array* obj = new DeviceFloat32Array(length);
            obj->Wrap(args.This());
            // Add properties here
            args.This()->Set(String::NewFromUtf8(isolate, "length"), Uint32::NewFromUnsigned(isolate, (uint32)obj->getLength()));
            args.GetReturnValue().Set(args.This());
            return;
        }
        else if (args[0]->IsFloat32Array())
        {
            Local<Float32Array> f32ah = Local<Float32Array>::Cast(args[0]);

            DeviceFloat32Array* obj = new DeviceFloat32Array(f32ah);
            obj->Wrap(args.This());
            // Add properties here
            args.This()->Set(String::NewFromUtf8(isolate, "length"), Uint32::NewFromUnsigned(isolate, (uint32)obj->getLength()));
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

// Explicitly reclaim device memory
void DeviceFloat32Array::destroy(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();
    DeviceFloat32Array* obj = ObjectWrap::Unwrap<DeviceFloat32Array>(args.Holder());
    jc_cuda::cudaFree_t(obj->m_ptrd);
    obj->m_ptrd = nullptr;
    obj->m_length = 0;
    args.GetReturnValue().Set(Undefined(isolate));
}

// Prototype function
//// return length
//void DeviceFloat32Array::length(const FunctionCallbackInfo<Value>& args)
//{
//    Isolate* isolate = args.GetIsolate();
//    DeviceFloat32Array* obj = ObjectWrap::Unwrap<DeviceFloat32Array>(args.Holder());
//    args.GetReturnValue().Set(Number::New(isolate, obj->getLength()));
//}

// copyFrom( host, offset_h, offset_d, size )
void DeviceFloat32Array::copyFrom(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();
    DeviceFloat32Array* f32ad = ObjectWrap::Unwrap<DeviceFloat32Array>(args.Holder());

    if (f32ad && args[0]->IsFloat32Array() && args[1]->IsUint32() && args[2]->IsUint32() && args[3]->IsUint32())
    {
        Local<Float32Array> f32ah = Local<Float32Array>::Cast(args[0]);
        uint32 ofsh   = args[1]->Uint32Value();
        uint32 ofsd   = args[2]->Uint32Value();
        uint32 size   = args[3]->Uint32Value();

        if (ofsh + size > f32ah->Length() || ofsd + size > f32ad->getLength())
        {
            isolate->ThrowException(Exception::RangeError(String::NewFromUtf8(isolate, jc::rangeError)));
            return;
        }

        jc_cuda::cudaMemcpyHostToDevice_t(f32ah->Buffer()->GetContents().Data()
            , f32ad->getData()
            , ofsh   * sizeof(float32)
            , ofsd   * sizeof(float32)
            , size   * sizeof(float32)
            );

        args.GetReturnValue().Set(args.This());
        return;
    }
    else
    {
        isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, jc::typArgError)));
        return;
    }
}

// copyTo( host, offset_h, offset_d, size )
void DeviceFloat32Array::copyTo(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();
    DeviceFloat32Array* f32ad = ObjectWrap::Unwrap<DeviceFloat32Array>(args.Holder());

    if (args[0]->IsFloat32Array() && args[1]->IsUint32() && args[2]->IsUint32() && args[3]->IsUint32())
    {
        Local<Float32Array> f32ah = Local<Float32Array>::Cast(args[0]);
        uint32 ofsh = args[1]->Uint32Value();
        uint32 ofsd = args[2]->Uint32Value();
        uint32 size = args[3]->Uint32Value();

        if (ofsh + size > f32ah->Length() || ofsd + size > f32ad->getLength())
        {
            isolate->ThrowException(Exception::RangeError(String::NewFromUtf8(isolate, jc::rangeError)));
            return;
        }

        jc_cuda::cudaMemcpyDeviceToHost_t(f32ad->getData()
            , f32ah->Buffer()->GetContents().Data()
            , ofsd   * sizeof(float32)
            , ofsh   * sizeof(float32)
            , size   * sizeof(float32)
            );

        args.GetReturnValue().Set(args.This());
        return;
    }
    else
    {
        isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, jc::typArgError)));
        return;
    }
}

// copy( src, offset_d, offset_s, size )
void DeviceFloat32Array::copy(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();
    DeviceFloat32Array* f32ad = ObjectWrap::Unwrap<DeviceFloat32Array>(args.Holder());
    DeviceFloat32Array* f32bd = ObjectWrap::Unwrap<DeviceFloat32Array>(args[0]->ToObject());

    if (f32bd && args[1]->IsUint32() && args[2]->IsUint32() && args[3]->IsUint32())
    {
        uint32 ofad = args[1]->Uint32Value();
        uint32 ofbs = args[2]->Uint32Value();
        uint32 size = args[3]->Uint32Value();

        if (ofad + size > f32ad->getLength() || ofbs + size > f32bd->getLength())
        {
            isolate->ThrowException(Exception::RangeError(String::NewFromUtf8(isolate, jc::rangeError)));
            return;
        }

        jc_cuda::cudaMemcpyDeviceToDevice_t(f32ad->getData()
            , f32bd->getData()
            , ofad   * sizeof(float32)
            , ofbs   * sizeof(float32)
            , size   * sizeof(float32)
            );

        args.GetReturnValue().Set(args.This());
        return;
    }
    else
    {
        isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, jc::typArgError)));
        return;
    }
}

// swap( v )
void DeviceFloat32Array::swap(const FunctionCallbackInfo<Value>& args)
{
    Isolate* isolate = args.GetIsolate();
    DeviceFloat32Array* f32ad = ObjectWrap::Unwrap<DeviceFloat32Array>(args.Holder());
    DeviceFloat32Array* f32bd = ObjectWrap::Unwrap<DeviceFloat32Array>(args[0]->ToObject());

    if (f32ad && f32bd)
    {
        size_t tmp = f32ad->m_length;
        f32ad->m_length = f32bd->m_length;
        f32bd->m_length = tmp;

        args.Holder()->Set(String::NewFromUtf8(isolate, "length"), Uint32::NewFromUnsigned(isolate, (uint32)f32ad->m_length));
        args[0]->ToObject()->Set(String::NewFromUtf8(isolate, "length"), Uint32::NewFromUnsigned(isolate, (uint32)f32bd->m_length));

        auto tmptr = f32ad->m_ptrd;
        f32ad->m_ptrd = f32bd->m_ptrd;
        f32bd->m_ptrd = tmptr;

        args.GetReturnValue().Set(args.This());
        return;
    }
    else
    {
        isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, jc::typArgError)));
        return;
    }
}


