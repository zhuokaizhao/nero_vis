/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#ifndef QIV_HAS_BEEN_INCLUDED
#define QIV_HAS_BEEN_INCLUDED

// begin includes
#include <stdio.h>  // for printf
#include <stdint.h> // for uint8_t
#include <stdlib.h> // for malloc, qsort
#include <assert.h> // for assert()
// define QIV_COMPILE when compiling libqiv or a non-Python thing depending on libqiv
#ifdef QIV_COMPILE
// things from Teem used in qiv API
#  include <teem/air.h>
#endif
// end includes

#ifdef __cplusplus
extern "C" {
#endif

// (much of this is copied from code of GLK's CMSC 23710 SciVis class)
/* Create typedef for "real" as either a "float" (single precision) or
   "double" (double precision) floating-point number, based on compile-time
   value of QIV_REAL_IS_DOUBLE. To change from real=float to real=double:

     make clean; make CFLAGS=-DQIV_REAL_IS_DOUBLE=1

*/
#ifndef QIV_REAL_IS_DOUBLE
#  define QIV_REAL_IS_DOUBLE 0
#endif
#if QIV_REAL_IS_DOUBLE
typedef double real;
#else
typedef float real;
#endif

/* "uint" is easier to type than any of C99's uint8_t, uint16_t, etc., and
   there is often no need for any specific size; just an "unsigned int" */
typedef unsigned int uint;
// also convenient to have
typedef unsigned char uchar;

// misc.c
extern void qivVerboseSet(int verb);
extern int qivVerboseGet(void);
extern const int qivRealIsDouble;
extern const char *qivBiffKey;
#define QIV qivBiffKey // identifies this library in biff error messages
extern real qivNan(unsigned short payload);

/*
  qivType: the pixel value types supported in qivImage
*/
typedef enum {
    qivTypeUnknown = 0, // (0) no type known
    qivTypeUChar,       // (1) unsigned char
    qivTypeReal,        // (2) real
} qivType;

/*
  The different kinds of integration we support; the enum values here are
  packaged into airEnum qivIntg_ae
*/
typedef enum {
    qivIntgUnknown = 0, // 0: don't know */
    qivIntgEuler,       // 1: Euler integration
    qivIntgMidpoint,    // 2: Midpoint method aka RK2
    qivIntgRK4,         // 3: Runge-Kutta fourth-order
} qivIntg;

/*
  The different kinds of vector field reconstruction we support
*/
typedef enum {
    qivKernUnknown = 0, // 0: don't know */
    qivKernBox,         // 1: box == nearest neighbor
    qivKernTent,        // 2: tent == bilinear
    qivKernCtmr,        // 3: Catmull-Rom
    qivKernBspln,       // 4: uniform cubic B-spline
} qivKern;

/*
  The streamline integration can proceed as long as certain conditions hold.
  This enum is for documenting the various reasons why one side of a
  streamline stopped (as recorded in qivSline->forwStop and
  qivSline->backStop), or why the streamline never went anywhere (as recorded
  in qivSline->seedStop).
*/
typedef enum {
    qivStopUnknown = 0, // 0: don't know
    qivStopNot,         /* 1: actually, not stopped (only for describing
                           seedpoint in qivSline->seedStop) */
    qivStopOutside,     // 2: convolution in vector field not possible
    qivStopNonfinite,   /* 3: convolution (or its subsequent normalization)
                           generated a non-finite vector field value (i.e. nan
                           or inf) */
    qivStopSteps,       /* 4: already took max number of steps in this
                           dir (only for describing streamlines ends in
                           qivSline->forwStop and qivSline->backStop) */
} qivStop;

/*
  struct qivArray: for all things on a 2D grid, both vector field and image data
*/
typedef struct qivArray_t {
    uint channel,   /* # samples on fastest (per-pixel) axis;
                       1 for scalar, 2 for vector, 3 for color */
      size0, size1; /* # samples on faster (size[0]), slower (size[1]) spatial
                       axis; these are the second and third axes in the
                       linearization; the fastest axis is one holding the *two*
                       vector components */
    real ItoW[9];   /* homogeneous coordinate mapping from index-space (faster
                       coordinate first) to the world-space in which the vector
                       components have been measured */
    qivType type;   /* type of the data; determines which of the union members
                        below to use */
    union {         /* union for the pointer to the image data; the pointer
                       values are all the same; this is just to avoid casting.
                       The right union member to use (data.uc vs data.rl)
                       determined at run-time by value of type */
        void *vd;
        uchar *uc;
        real *rl;
    } data;
} qivArray;

/*
  struct qivCtx: all the state associated with computing convolution, and
  streamlines, and LIC (line integral convolution). Mostly implemented in
  ctx.c.

  Note that nothing in this project is intended to run multi-threaded; in this
  sense convolution in this project is more like that in p2mapr than p3rendr.
*/
typedef struct qivCtx_t {
    const qivArray *qar; /* vector field to process. */
    qivKern kern;        /* how do we interpolate */
    real WtoI[9];        /* inverse of vfl->ItoW, for homog coord mapping
                            from world to index space */
    int insideOnly;      /* do not actually care about the full convolution
                            result, only the value of "inside" */

    /* output to be set in qivConvoEval:
       inside: the following convolution results could be computed
       vec: the vector reconstructed componentwise by convolution
    */
    int inside;
    real vec[2];
} qivCtx;

/*
  qivSline: A container for a single streamline. qivSlineTrace uses this to
  store its results. "halfLen" determines for how many positions the "pos"
  array is allocated (as detailed below), and all the remaining fields are set
  by qivSlineTrace.  Implemented in sline.c.
*/
typedef struct qivSline_t {
    real seed[2];               /* where this was seeded */
    uint halfLen;               /* pos is allocated for 2*(1 + 2*halfLen) reals, i.e.
                                   for (1 + 2*halfLen) 2-vectors of coordinates.
                                   pos+2*halfLen is a 2-vector giving the position of
                                   the starting (seed) point of the streamline.  Even if
                                   halfLen is 0, pos can store a position, in which case
                                   this struct is just a way of storing a vector at some
                                   location */
    real *pos;                  /* world-space coords of all points along streamline, as
                                   a 2 (fast) by 1+2*halfLen (slow) array.  This is NULL
                                   upon return from qivSlineNew() */
    qivStop seedStop;           /* why streamline integration couldn't go anywhere from
                                   the seedpoint. Value qivStopSteps isn't used. Value
                                   qivStopNot means "we started inside the field, so
                                   integration may have gone ok". In that case, at least
                                   the 2-vector at pos+2*halfLen is set (to the location
                                   of the seed point), and more positions are set in pos
                                   according to how long the integration could proceed
                                   forward and backward, as determined by
                                   qivSlineTrace(), and as recorded in forwNum,
                                   forwStop, backNum, and backStop. */
    uint forwNum,               /* the forward (or "downstream", following the vectors)
                                   part of streamline is from pos+2*halfLen (the seed
                                   point) to pos+2*(halfLen+forwNum) (the head of the
                                   streamline, imagined as an arrow). Working correctly,
                                   qivSlineTrace() ensures 0 <= forwNum <= halfLen. */
      backNum;                  /* the backward (or "upstream", following the negation
                                   of the vectors) part of streamline is from
                                   pos+2*(halfLen-backNum) (the tail of the streamline,
                                   considered as an arrow) to pos+2*halfLen (the
                                   seedpoint). Working correctly, qivSlineTrace()
                                   ensures 0 <= backNum <= halfLen. */
    qivStop forwStop, backStop; /* why integration stopped in the forward and backward
                                   directions of the streamline. Upon completion of
                                   qivSlineTrace these should be set to something
                                   besides qivStopNot (all streamlines must stop
                                   eventually, and stop for some specific reason). */
    real vecSeed[2];            // vector (reconstructed from vector field) at seed point
} qivSline;

// aenum.c: airEnums
extern const airEnum *const qivType_ae;
extern const airEnum *const qivIntg_ae;
extern const airEnum *const qivStop_ae;
extern const airEnum *const qivKern_ae;

// array.c
extern qivArray *qivArrayNew(void);
extern int qivArrayAlloc(qivArray *qar, uint channel, uint size0, uint size1,
                         qivType dtype);
extern int qivArraySet(qivArray *qar, uint channel, uint size0, uint size1,
                       qivType dstType, const void *srcData, int srcNType,
                       const double *edge0, const double *edge1, const double *orig);
extern qivArray *qivArrayNix(qivArray *qar);
extern int qivArraySave(const char *fname, const qivArray *qar);

// ctx.c: for setting up and using the qivCtx
extern qivCtx *qivCtxNew(const qivArray *qar, qivKern kern);
extern qivCtx *qivCtxNix(qivCtx *ctx);

// convo.c: for convolution
extern void qivConvoEval(qivCtx *ctx, real xw, real yw, int sgn, int norm);

// sline.c: for storing and computing streamlines
extern qivSline *qivSlineNew(void);
extern int qivSlineAlloc(qivSline *sln, uint halfLen);
extern qivSline *qivSlineNix(qivSline *sln);
extern int qivSlineTrace(qivSline *const sln, real seedX, real seedY, uint halfLen,
                         real hh, int normalize, int intg, qivCtx *ctx);
extern void qivSlinePrint(qivSline *const sln);

// lic.c: for Line Integral Convolution: TODO finish qivLICEval
/*
extern int qivLICEval(real *const result, qivSline *const sln, real wx, real wy,
                      uint halfLen, real hh, int normalize, int intg,
                      const qivField *rnd, const real *rWtoI, int rndLinterp,
                      qivCtx *ctx);
extern int qivLIC(qivField *const lmg, qivField *const pmg, int prop, uint halfLen,
                  real hh, int normalize, int intg, const qivField *rnd, int
                  rndLinterp, qivCtx *ctx);
*/

#ifdef __cplusplus
}
#endif
#endif // QIV_HAS_BEEN_INCLUDED
