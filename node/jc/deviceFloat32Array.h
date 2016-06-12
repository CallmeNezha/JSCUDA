#ifndef __DEVICEFLOAT32ARRAY_H__
#define __DEVICEFLOAT32ARRAY_H__

#include <node.h>
#include <node_object_wrap.h>
#include "jc_parameter.h"

using namespace v8;


class DeviceFloat32Array final : public node::ObjectWrap
{
private:
    float*      m_ptrd;   // point to device memory
    size_t      m_length; // length of array

    static Persistent<Function> constructor;

public:
    static void Init(Local<Object> exports);

    // C++ functions
    size_t      getLength() const{ return m_length; }

private:
    explicit DeviceFloat32Array(unsigned int length);
    explicit DeviceFloat32Array(const Local<Float32Array>& f32a);
    ~DeviceFloat32Array();

    // New keyword
    static void New(const FunctionCallbackInfo<Value>& args);

    //Prototype Function
    static void length(const FunctionCallbackInfo<Value>& args);
    static void copyFrom(const FunctionCallbackInfo<Value>& args);
    static void copyTo(const FunctionCallbackInfo<Value>& args);
};


#endif //!__DEVICEFLOAT32ARRAY_H__