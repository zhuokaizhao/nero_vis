/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#include "qiv.h"
#include "qivPrivate.h"

qivField *
qivFieldNew() {
    qivField *ret;
    ret = MALLOC(1, qivField);
    assert(ret);
    ret->data = NULL;
    ret->size0 = 0;
    ret->size1 = 0;
    M3_SET_NAN(ret->ItoW);
    ret->data = NULL;
    return ret;
}

static int
_qivFieldAlloc(qivField *vfl, uint size0, uint size1) {
    assert(vfl);
    if (vfl->size0 * vfl->size1 != size0 * size1) {
        free(vfl->data);
        vfl->data = NULL;
        vfl->data = MALLOC(2 * size0 * size1, real);
        if (!vfl->data) {
            biffAddf(QIV, "%s: failed to allocate 2 * %u x %u field data array",
                     __func__, size0, size1);
            return 1;
        }
        vfl->size0 = size0;
        vfl->size1 = size1;
    }
    return 0;
}

int
qivFieldSet(qivField *vfl, uint size0, uint size1, const double *edge0,
            const double *edge1, const double *orig, void *data, int ntype) {
    if (!vfl) {
        biffAddf(QIV, "%s: got NULL field", __func__);
        return 1;
    }
    if (!(size0 >= 8 && size1 >= 8)) {
        biffAddf(QIV, "%s: sizes %u %u not valid", __func__, size0, size1);
        return 1;
    }
    /* edge0, edge1, orig: either all NULL or non-NULL */
    uint haveP = !!edge0 + !!edge1 + !!orig;
    if (!(0 == haveP || 3 == haveP)) {
        biffAddf(QIV,
                 "%s: edge0 (%p), edge1 (%p), orig (%p) should either be "
                 "all non-NULL or all NULL",
                 __func__, CVOIDP(edge0), CVOIDP(edge1), CVOIDP(orig));
        return 1;
    }
    if (!data) {
        biffAddf(QIV, "%s: got NULL data", __func__);
        return 1;
    }
    if (airEnumValCheck(nrrdType, ntype)) {
        biffAddf(QIV, "%s: got invalid ntype %d", __func__, ntype);
        return 1;
    }
    if (!(nrrdTypeFloat == ntype || nrrdTypeDouble == ntype)) {
        biffAddf(QIV, "%s: got ntype %s (%d) but need %s (%d) or %s (%d)", __func__,
                 airEnumStr(nrrdType, ntype), ntype,                 /* */
                 airEnumStr(nrrdType, nrrdTypeFloat), nrrdTypeFloat, /* */
                 airEnumStr(nrrdType, nrrdTypeDouble), nrrdTypeDouble);
        return 1;
    }
    /* end of error checking */

    if (edge0) {                            /* and edge1 and orig */
        M3_SET(vfl->ItoW,                   /* */
               edge0[0], edge1[0], orig[0], /* */
               edge0[1], edge1[1], orig[1], /* */
               0, 0, 1);
    } else {
        M3_SET(vfl->ItoW, /* */
               1, 0, 0,   /* */
               0, 1, 0,   /* */
               0, 0, 1);
    }
    if (_qivFieldAlloc(vfl, size0, size1)) {
        biffAddf(QIV, "%s: trouble allocating field", __func__);
        return 1;
    }
    uint N = 2 * size0 * size1;
    real *dd = vfl->data;
    if (nrrdTypeFloat == ntype) {
        for (uint I = 0; I < N; I++) {
            dd[I] = (real)((float *)data)[I];
        }
    } else { /* nrrdTypeDouble == ntype */
        for (uint I = 0; I < N; I++) {
            dd[I] = (real)((double *)data)[I];
        }
    }
    return 0;
}

qivField *
qivFieldNix(qivField *vfl) {
    if (vfl) {
        free(vfl->data);
        free(vfl);
    }
    return NULL;
}
