#ifndef __JC_PARAMTER_H__
#define __JC_PARAMTER_H__

#include "jc_api.h"

// Parameters
#define DEBUG_OUTPUT_ON  0


// Type alias
typedef float           float32;
typedef double          float64;
typedef unsigned int    uint32;
typedef int             int32;


// Global extern
extern bool                    jcg_bCudaInitialized;
extern jc_cuda::cublasHandle_t jcg_cublasHandle;




#endif //!__JC_PARAMTER_H__
