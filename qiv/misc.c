/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#include "qiv.h"
#include "qivPrivate.h"

#ifndef NDEBUG
// still a global, but no longer part of API, because CFFI didn't seem to correctly
// handle "qivVerbose = 4", at least not after "from qiv import *"
int _qivVerbose = 0;
#endif

void
qivVerboseSet(int verb) {
#ifndef NDEBUG
    _qivVerbose = verb;
#else
    (void)(verb);
    fprintf(stderr, "!!!\n!!! %s: not available because compiled with -NDEBUG\n!!!\n",
            __func__);
#endif
}
int
qivVerboseGet() {
    return _qivVerbose;
}

#if QIV_REAL_IS_DOUBLE
const int qivRealIsDouble = 1;
#else
const int qivRealIsDouble = 0;
#endif

const char *qivBiffKey = "qiv";

typedef union {
#if QIV_REAL_IS_DOUBLE
    // for accessing bits of a 64-bit double
    uint64_t i;
    double v;
#else
    // for accessing bits of a 32-bit float
    uint32_t i;
    float v;
#endif
} Real;

real
qivNan(unsigned short payload) {
    Real rr;
    /* same logic for both meanings of real: make a non-finite number by setting all the
       exponent bits, make it a NaN by making sure highest bit of fraction is on (else
       it would be an infinity), and then put the 16-bit payload in the lowest bits. */
#if QIV_REAL_IS_DOUBLE
    rr.i = ((uint64_t)0x7ff << 52) | ((uint64_t)1 << 51) | ((uint64_t)payload);
#else
    rr.i = ((uint32_t)0xff << 23) | ((uint32_t)1 << 22) | ((uint32_t)payload);
#endif
    return rr.v;
}
