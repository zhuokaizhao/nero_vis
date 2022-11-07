/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#ifdef __cplusplus
extern "C" {
#endif

/* utility macro for malloc() (memory allocation on the heap) for N things of type T, and
   casting to T* type */
#define MALLOC(N, T) (T *)(malloc((N) * sizeof(T)))

#define CVOIDP(x) ((const void *)(x))

/* The things #define'd here according to SCIVIS_REAL_IS_DOUBLE simplify connecting to
   other external libraries that do not know about our choice of "real", but which can
   handle either float or double. */
#if QIV_REAL_IS_DOUBLE
#  define airTypeReal  airTypeDouble
#  define nrrdTypeReal nrrdTypeDouble
#else
#  define airTypeReal  airTypeFloat
#  define nrrdTypeReal nrrdTypeFloat
#endif

#include <assert.h> // for assert()
#include <math.h>
#include "qivMath.h"

#ifdef __cplusplus
}
#endif
