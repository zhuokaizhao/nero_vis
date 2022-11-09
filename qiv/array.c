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
    ret->dtype = qivTypeUnknown;
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
_metaDataCheck(uint channel, uint size0, uint size1, qivType dtype, const double *edge0,
               const double *edge1, const double *orig) {
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
    uint haveP = !!edge0 + !!edge1 + !!orig;
    if (!(0 == haveP || 3 == haveP)) {
        biffAddf(QIV,
                 "%s: edge0 (%p), edge1 (%p), orig (%p) should either be "
                 "all non-NULL or all NULL",
                 __func__, CVOIDP(edge0), CVOIDP(edge1), CVOIDP(orig));
        return 1;
    }
    if (haveP) { // have all of edge0, edge1, orig
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

int
qivArrayAlloc(qivArray *qar, uint channel, uint size0, uint size1, qivType dtype) {
    if (!qar) {
        biffAddf(QIV, "%s: got NULL array pointer", __func__);
        return 1;
    }
    if (_metaDataCheck(channel, size0, size1, dtype, NULL, NULL, NULL)) {
        biffAddf(QIV, "%s: problem with basic meta-data", __func__);
        return 1;
    }
    int doalloc;
    if (!(qar->data.vd)) {
        // definitely not already allocated
        doalloc = 1;
    } else if (qar->channel != channel || qar->size0 != size0 || qar->size1 != size1
               || qar->dtype != dtype) {
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
    // qar->ItoW untouched
    qar->channel = channel;
    qar->size0 = size0;
    qar->size1 = size1;
    qar->dtype = dtype;
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
   */
int
qivArraySet(qivArray *qar, uint channel, uint size0, uint size1, qivType dstType,
            const void *srcData, int srcNType, const double *edge0, const double *edge1,
            const double *orig) {
    if (!qar) {
        biffAddf(QIV, "%s: got NULL array", __func__);
        return 1;
    }
    if (_metaDataCheck(channel, size0, size1, dstType, edge0, edge1, orig)) {
        biffAddf(QIV, "%s: problem with meta-data", __func__);
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
    if (edge0) {                            /* and edge1 and orig */
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
    return 0;
}
