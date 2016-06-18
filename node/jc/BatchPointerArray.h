#ifndef __BATCHPOINTERARRAY_H__
#define __BATCHPOINTERARRAY_H__

#include <node.h>
#include <node_object_wrap.h>
#include "jc_parameter.h"

using namespace v8;


class BatchPointerArray final : public node::ObjectWrap
{
public:
    enum   PointerType{
        Float32_Pointer = 0,
        Type_Count
    };

private:
    void*       m_ptrd;   // Point to device memory
    size_t      m_size;   // Size of array in bytes
    PointerType m_type;

    static Persistent<Function> constructor;

public:
    static void Init(Local<Object> exports);

    // C++ functions
    size_t      getLength()     const{ return m_size / getTypeSize(); }
    void*       getData()            { return m_ptrd; }
    PointerType getType()       const{ return m_type; }
    size_t      getTypeSize()   const;

private:
    explicit BatchPointerArray(size_t size, PointerType type, void *ptrd);
    BatchPointerArray(const BatchPointerArray &) = delete;
    BatchPointerArray& operator = (const BatchPointerArray &) = delete;
    ~BatchPointerArray();

    // New keyword
    static void New(const FunctionCallbackInfo<Value>& args);

    // Prototype Function
    //static void length  (const FunctionCallbackInfo<Value>& args);
    static void destroy(const FunctionCallbackInfo<Value>& args);
};


#endif //!__DEVICEFLOAT32ARRAY_H__