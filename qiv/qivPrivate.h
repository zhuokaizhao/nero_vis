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
   handle either float or double. Note these things do not start with the "qiv" prefix,
   which is misleading, but that's why these are limited to the private header, with the
   assumption that qiv coders are grateful for the familiar yet more general names. */
#if QIV_REAL_IS_DOUBLE
#  define airTypeReal  airTypeDouble
#  define nrrdTypeReal nrrdTypeDouble
#  define ell_3m_print ell_3m_print_d
#else
#  define airTypeReal  airTypeFloat
#  define nrrdTypeReal nrrdTypeFloat
#  define ell_3m_print ell_3m_print_f
#endif

#include <assert.h> // for assert()
#include <math.h>
#include <tgmath.h> // type-generic math macros
/* <tgmath.h> #includes <complex.h>, which in turn defines a macro "I", which removes "I"
   from your possible variable names (using it creates cryptic compiler error messages).
   We're allowed to undefine "I", so we do. */
#undef I

// things from Teem used in implementation, but not part of qiv API
#include <teem/biff.h>
#include <teem/hest.h>
#include <teem/nrrd.h>
#include <teem/ell.h>

#include "qivMath.h"

// misc.c
#ifndef NDEBUG
extern int _qivVerbose;
#else
#  define Verbose 0
#endif
extern void _qiv3M_aff_inv(real inv[9], const real m[9]);

#ifdef __cplusplus
}
#endif
