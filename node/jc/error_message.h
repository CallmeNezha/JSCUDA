#ifndef __ERROR_MESSAGE_H__
#define __ERROR_MESSAGE_H__

namespace jc
{
    static char* rangeError  = "Range Error: Out of range";
    static char* rangeMError = "Range Error: Range doesn't match";

    static char* numArgError = "Argument Error: Wrong number of arguments";
    static char* typArgError = "Argument Error: Wrong type of arguments";

    static char* deviceError = "Device Error: CUDA device not initialized!";
    static char* cublasError = "cuBLAS Error: cuBLAS not initialized!";
    static char* noNewError  = "Syntax Error: Lack of 'new' keyword";

}



#endif //!__ERROR_MESSAGE_H__