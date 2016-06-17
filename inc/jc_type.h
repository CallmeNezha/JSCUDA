#ifndef __JC_TYPE_H__
#define __JC_TYPE_H__

#ifndef UINT_T
typedef unsigned int        uint;
#endif // !UINT_T

enum ErrorType
{
    JC_SUCCESS = 0,
    JC_PARAM_ERROR,
    JC_ERROR_COUNT
};

typedef struct
{
    unsigned int numRow;
    unsigned int numCol;
    float*       elements;
    bool         transposed;
    // In column major
} Matrix;

typedef struct
{
    unsigned int numRow;
    unsigned int numCol;
    float**      elementsArray;
    bool         transposed;
    unsigned int count;
    // In column major
} MatrixBatch;

typedef struct  
{
    unsigned int length;
    float*       elements;
} Vector;

#endif //!__JC_TYPE_H__