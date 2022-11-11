/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#include "qiv.h"
#include "qivPrivate.h"

static size_t
_typeSize(qivType dtype) {
    size_t ret;
    switch (dtype) {
    case qivTypeUChar:
        ret = sizeof(unsigned char);
        break;
    case qivTypeReal:
        ret = sizeof(real);
        break;
    default:
        ret = 0;
        break;
    }
    return ret;
}

qivArray *
qivArrayNew() {
    qivArray *ret = MALLOC(1, qivArray);
    assert(ret);
    ret->channel = ret->size0 = ret->size1 = 0;
    M3_SET_NAN(ret->ItoW);
    M3_SET_NAN(ret->WtoI);
    ret->type = qivTypeUnknown;
    ret->data.vd = NULL;
    return ret;
}

qivArray *
qivArrayNix(qivArray *qar) {
    if (qar) {
        free(qar->data.vd);
        free(qar);
    }
    return NULL;
}

static int
shapeTypeCheck(uint channel, uint size0, uint size1, qivType dtype) {
    if (!(1 <= channel && channel <= 3)) {
        biffAddf(QIV, "%s: invalid channel value %u", __func__, channel);
        return 1;
    }
    if (!(5 <= size0 && 5 <= size1)) {
        biffAddf(QIV, "%s: array sizes (%u,%u) must be >= 5", __func__, size0, size1);
        return 1;
    }
    if (airEnumValCheck(qivType_ae, dtype)) {
        biffAddf(QIV, "%s: invalid type %d", __func__, dtype);
        return 1;
    }
    return 0;
}

static int
orientationCheck(const double *edge0, const double *edge1, const double *orig) {
    uint haveP = !!edge0 + !!edge1 + !!orig;
    if (!(0 == haveP || 3 == haveP)) {
        biffAddf(
          QIV,
          "%s: edge0 (%p), edge1 (%p), orig (%p) should be all NULL, or all non-NULL",
          __func__, CVOIDP(edge0), CVOIDP(edge1), CVOIDP(orig));
        return 1;
    }
    if (haveP) {
        int isf[3] = {V2_ISFINITE(edge0), V2_ISFINITE(edge1), V2_ISFINITE(orig)};
        if (!(isf[0] && isf[1] && isf[2])) {
            biffAddf(QIV, "%s: edge0, edge1, orig must all be finite (got %d, %d, %d)",
                     __func__, isf[0], isf[1], isf[2]);
            return 1;
        }
        real el[2] = {V2_LEN(edge0), V2_LEN(edge1)};
        if (!(el[0] && el[1])) {
            biffAddf(QIV, "%s: edge0 and edge1 must be non-zero (got len %g, %g)",
                     __func__, el[0], el[1]);
            return 1;
        }
    }
    return 0;
}

static void
orientationSet(qivArray *qar, const double *edge0, const double *edge1,
               const double *orig) {
    if (edge0 && edge1 && orig) {
        M3_SET(qar->ItoW,                   /* */
               edge0[0], edge1[0], orig[0], /* */
               edge0[1], edge1[1], orig[1], /* */
               0, 0, 1);
    } else {
        M3_SET(qar->ItoW, /* */
               1, 0, 0,   /* */
               0, 1, 0,   /* */
               0, 0, 1);
    }
    _qiv3M_aff_inv(qar->WtoI, qar->ItoW);
    return;
}

int
qivArrayAlloc(qivArray *qar, uint channel, uint size0, uint size1, qivType dtype) {
    if (!qar) {
        biffAddf(QIV, "%s: got NULL array pointer", __func__);
        return 1;
    }
    if (shapeTypeCheck(channel, size0, size1, dtype)) {
        biffAddf(QIV, "%s: problem with shape or type", __func__);
        return 1;
    }
    int doalloc;
    if (!(qar->data.vd)) {
        // definitely not already allocated
        doalloc = 1;
    } else if (qar->channel != channel || qar->size0 != size0 || qar->size1 != size1
               || qar->type != dtype) {
        // already allocated, but not the right size/type
        free(qar->data.vd);
        doalloc = 1;
    } else {
        // cool; re-use existing allocating
        doalloc = 0;
    }
    if (doalloc) {
        qar->data.vd = malloc(channel * size0 * size1 * _typeSize(dtype));
        if (!qar->data.vd) {
            biffAddf(QIV, "%s: failed to allocate %u * %u * %u %s array data", __func__,
                     channel, size0, size1, airEnumStr(qivType_ae, dtype));
            return 1;
        }
    }
    /* we invalidate ItoW and WtoI every time, even if !doalloc, because the logical
       result of calling an allocation function is that something has changed about index
       space, which means ItoW has to be re-set, with qivArrayOrientationSet() */
    M3_SET_NAN(qar->ItoW);
    M3_SET_NAN(qar->WtoI);
    qar->channel = channel;
    qar->size0 = size0;
    qar->size1 = size1;
    qar->type = dtype;
    return 0;
}

/* wrapper around qivArrayAlloc that also sets contents
   qar: result of qivArrayNew()
   channel: # values per pixel; 2 for vector field, 1 or 3 for images
   size0, size1: # samples along faster, slower spatial axes
   dstType: the intended (destination) data type inside qar:
        qivTypeUChar (for images) or qivTypeReal (for vector fields)
   srcData: where to copy data values (this pointer this is not saved)
   srcNType: the type of the values in srcData, from the nrrdType enum
       nrrdTypeUChar (for images) or nrrdTypeFloat, nrrdTypeDouble for vector data
   edge0,edge1: 2-vectors giving change in world-space from increasing the
       faster,slower spatial array index; will go into first,second columns of ItoW
   orig: location of first sample in world-sapce, will go into third column of ItoW
   As a little convenience, all of edge0, edge1, orig can be NULL to say:
      "make ItoW be the identity matrix"
   */
int
qivArraySet(qivArray *qar, uint channel, uint size0, uint size1, qivType dstType,
            const void *srcData, int srcNType, const double *edge0, const double *edge1,
            const double *orig) {
    if (!qar) {
        biffAddf(QIV, "%s: got NULL array", __func__);
        return 1;
    }
    if (shapeTypeCheck(channel, size0, size1, dstType)) {
        biffAddf(QIV, "%s: problem with shape or type", __func__);
        return 1;
    }
    if (orientationCheck(edge0, edge1, orig)) {
        biffAddf(QIV, "%s: problem with orientation (edge0, edge1, orig)", __func__);
        return 1;
    }
    if (!srcData) {
        biffAddf(QIV, "%s: got NULL data", __func__);
        return 1;
    }
    if (airEnumValCheck(nrrdType, srcNType)) {
        biffAddf(QIV, "%s: got invalid source (nrrdType) data type %d", __func__,
                 srcNType);
        return 1;
    }
    if (!(nrrdTypeUChar == srcNType || nrrdTypeFloat == srcNType
          || nrrdTypeDouble == srcNType)) {
        biffAddf(QIV, "%s: got srcNType %s (%d) but need %s (%d), %s (%d), or %s (%d)",
                 __func__, airEnumStr(nrrdType, srcNType), srcNType, /* */
                 airEnumStr(nrrdType, nrrdTypeUChar), nrrdTypeUChar, /* */
                 airEnumStr(nrrdType, nrrdTypeFloat), nrrdTypeFloat, /* */
                 airEnumStr(nrrdType, nrrdTypeDouble), nrrdTypeDouble);
        return 1;
    }
    switch (dstType) {
    case qivTypeUChar:
        if (nrrdTypeUChar != srcNType) {
            biffAddf(QIV, "%s: sorry, need srcNType %s for dstType %s", __func__,
                     airEnumStr(nrrdType, nrrdTypeUChar),
                     airEnumStr(qivType_ae, qivTypeUChar));
            return 1;
        }
        break;
    case qivTypeReal:
        if (!(nrrdTypeFloat == srcNType || nrrdTypeDouble == srcNType)) {
            biffAddf(QIV, "%s: sorry, need srcNType %s or %s for dstType %s", __func__,
                     airEnumStr(nrrdType, nrrdTypeFloat),
                     airEnumStr(nrrdType, nrrdTypeDouble),
                     airEnumStr(qivType_ae, qivTypeReal));
            return 1;
        }
        break;
    default:
        biffAddf(QIV, "%s: sorry, dstType %s (%d) not implemented", __func__,
                 airEnumStr(qivType_ae, dstType), dstType);
        return 1;
    }

    /* error checking of arguments done; now allocate and set output
       (the next call will trigger a slightly redundant metadatacheck) */
    if (qivArrayAlloc(qar, channel, size0, size1, dstType)) {
        biffAddf(QIV, "%s: trouble allocating output", __func__);
        return 1;
    }
    uint N = channel * size0 * size1;
    // sd, dd = source, destination data pointers
    uchar *dd_uc = qar->data.uc;
    real *dd_rl = qar->data.rl;
    const uchar *sd_uc = (const uchar *)srcData;
    const float *sd_fl = (const float *)srcData;
    const double *sd_db = (const double *)srcData;
    /* the simplicity of this code depends on the constraints imposed by the error
       checking above */
    if (qivTypeUChar == dstType) {
        memcpy(dd_uc, sd_uc, N); // sizeof(uchar) == 1
    } else {
        /* qivTypeReal == dstType; one of these can't be a memcpy, so don't bother making
           either one a memcpy (unless profiling reveals it to be a bottleneck) */
        if (nrrdTypeFloat == srcNType) {
            for (uint I = 0; I < N; I++) {
                dd_rl[I] = (real)sd_fl[I];
            }
        } else { /* nrrdTypeDouble == srcNType */
            for (uint I = 0; I < N; I++) {
                dd_rl[I] = (real)sd_db[I];
            }
        }
    }
    orientationSet(qar, edge0, edge1, orig);
    return 0;
}

int
qivArrayOrientationSet(qivArray *qar, const double *edge0, const double *edge1,
                       const double *orig) {
    if (orientationCheck(edge0, edge1, orig)) {
        biffAddf(QIV, "%s: problem with orientation (edge0, edge1, orig)", __func__);
        return 1;
    }
    orientationSet(qar, edge0, edge1, orig);
    return 0;
}

int
qivArraySyntheticFlowSet(qivArray *qar, const real v0, const real v1, const real j00,
                         const real j01, const real j10, const real j11) {
    if (!qar) {
        biffAddf(QIV, "%s: got NULL pointer", __func__);
        return 1;
    }
    if (!(2 == qar->channel && qivTypeReal == qar->type)) {
        biffAddf(QIV, "%s: need 2-channel %s array (not %u-channel %s)", __func__,
                 airEnumStr(qivType_ae, qivTypeReal), /* */
                 qar->channel, airEnumStr(qivType_ae, qar->type));
        return 1;
    }
    if (!M3_ISFINITE(qar->ItoW)) {
        biffAddf(QIV, "%s: array does not seem to have ItoW set", __func__);
        return 1;
    }
    if (!(isfinite(v0) && isfinite(v1) && /* */
          isfinite(j00) && isfinite(j01) && isfinite(j10) && isfinite(j11))) {
        biffAddf(QIV,
                 "%s: not all of vector (%g,%g) or jacobian(%g,%g,%g,%g) "
                 "components exist ",
                 __func__, v0, v1, j00, j01, j10, j11);
        return 1;
    }
    uint sz0 = qar->size0, sz1 = qar->size1;
    real pi[3], pw[3],             // homog-coord position in index,world space
      dv[2],                       // change in vector due to offset from origin
      J[4] = {j00, j01, j10, j11}; // Jacobian
    real *vv = qar->data.rl;       // pointer to current vector
    for (uint jj = 0; jj < sz1; jj++) {
        for (uint ii = 0; ii < sz0; ii++) {
            V3_SET(pi, ii, jj, 1);
            MV3_MUL(pw, qar->ItoW, pi); // index to world
            MV2_MUL(dv, J, pw);         // world position to vector change
            V2_SET(vv, v0 + dv[0], v1 + dv[1]);
            vv += 2; // move to next vector
        }
    }
    return 0;
}

/* creates a Nrrd around a given qivArray
   NOTE: this asserts some things that are not assumptions of qiv itself:
   -- that world-space is nrrdSpaceRightUp,
   -- that the axes are cell-centered
   These assumptions make the nrrd compatible with SciVis class p4vectr
*/
static Nrrd *
_nrrdWrapper(const qivArray *qar) {
    int ntype;
    switch (qar->type) {
    case qivTypeUChar:
        ntype = nrrdTypeUChar;
        break;
    case qivTypeReal:
        ntype = nrrdTypeReal;
        break;
    default:
        biffAddf("%s: qar->type %s (%d) not handled", __func__,
                 airEnumStr(qivType_ae, qar->type), qar->type);
        return NULL;
    }
    uint dim;
    size_t size[3];
    if (1 == qar->channel) {
        dim = 2;
        size[0] = qar->size0;
        size[1] = qar->size1;
    } else {
        dim = 3;
        size[0] = qar->channel;
        size[1] = qar->size0;
        size[2] = qar->size1;
    }
    // error checking done
    Nrrd *ret = nrrdNew();
    if (nrrdWrap_nva(ret, qar->data.vd, ntype, dim, size)
        || nrrdSpaceSet(ret, nrrdSpaceRightUp)) {
        biffMovef(QIV, NRRD, "%s: failed to wrap nrrd", __func__);
        nrrdNix(ret);
        return NULL;
    }
    // dim==2 --> dim-2, dim-1 == 0, 1
    // dim==3 --> dim-2, dim-1 == 1, 2
    // ItoW:
    // 0  1  2
    // 3  4  5
    ELL_2V_SET(ret->axis[dim - 2].spaceDirection, qar->ItoW[0], qar->ItoW[3]);
    ELL_2V_SET(ret->axis[dim - 1].spaceDirection, qar->ItoW[1], qar->ItoW[4]);
    ELL_2V_SET(ret->spaceOrigin, qar->ItoW[2], qar->ItoW[5]);
    ret->axis[dim - 2].center = nrrdCenterCell;
    ret->axis[dim - 1].center = nrrdCenterCell;
    return ret;
}

int
qivArraySave(const char *fname, const qivArray *qar) {
    if (!(fname && qar)) {
        biffAddf("%s: got NULL pointer (%p,%p)", __func__, CVOIDP(fname), CVOIDP(qar));
        return 1;
    }
    Nrrd *nrd = _nrrdWrapper(qar);
    if (!nrd) {
        biffAddf(QIV, "%s: trouble wrapping", __func__);
        return 1;
    }
    NrrdIoState *nio = nrrdIoStateNew();
    nio->bareText = AIR_FALSE;
    nio->moreThanFloatInText = AIR_FALSE;
    if (nrrdSave(fname, nrd, nio)) {
        biffMovef(QIV, NRRD, "%s: trouble saving", __func__);
        nrrdNix(nrd);
        nrrdIoStateNix(nio);
        return 1;
    }
    nrrdNix(nrd);
    nrrdIoStateNix(nio);
    return 0;
}

int
qivArrayBBox(double xyMin[2], double xyMax[2], const qivArray *qar) {
    if (!(xyMin && xyMax && qar)) {
        biffAddf(QIV, "%s: got NULL pointer", __func__);
        return 1;
    }
    if (!M3_ISFINITE(qar->ItoW)) {
        biffAddf(QIV, "%s: array does not seem to have ItoW set", __func__);
        return 1;
    }
    real pi[3] = {0, 0, 1}, pw[3] = {0, 0, 1};
    uint sz0 = qar->size0;
    uint sz1 = qar->size1;
    for (uint jj = 0; jj < 2; jj++) {
        pi[1] = jj * (sz1 - 1);
        for (uint ii = 0; ii < 2; ii++) {
            pi[0] = ii * (sz0 - 1);
            MV3_MUL(pw, qar->ItoW, pi); // index to world
            if (!ii && !jj) {
                /* initialize min, max corners */
                V2_COPY(xyMin, pw);
                V2_COPY(xyMax, pw);
            } else {
                /* update corners */
                xyMin[0] = MIN(xyMin[0], pw[0]);
                xyMin[1] = MIN(xyMin[1], pw[1]);
                xyMax[0] = MAX(xyMax[0], pw[0]);
                xyMax[1] = MAX(xyMax[1], pw[1]);
            }
        }
    }
    return 0;
}
