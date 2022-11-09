/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#include "qiv.h"
#include "qivPrivate.h"

qivCtx *
qivCtxNew(const qivField *vfl, qivKern kern) {
    if (!vfl) {
        biffAddf(QIV, "%s: got NULL pointer", __func__);
        return NULL;
    }
    if (airEnumValCheck(qivKern_ae, kern)) {
        biffAddf(QIV, "%s: got invalid kern %d", __func__, kern);
        return NULL;
    }
    qivCtx *ctx = MALLOC(1, qivCtx);
    assert(ctx);
    ctx->vfl = vfl;
    ctx->kern = kern;
    const real *m = vfl->ItoW;
    /* clang-format off */
    real aa = m[0]; real bb = m[1]; real rr = m[2];
    real cc = m[3]; real dd = m[4]; real ss = m[5];
    real det = bb*cc - aa*dd;
    M3_SET(ctx->WtoI,
           -dd/det, bb/det, (dd*rr - bb*ss)/det,
           cc/det, -aa/det, (-cc*rr + aa*ss)/det,
           0, 0, 1);
    /* clang-format on */
    if (_qivVerbose > 2) {
        real test[9];
        M3_MUL(test, ctx->WtoI, vfl->ItoW);
        printf("WtoI * ItoW = \n");
        ell_3m_print(stdout, test);
    }
    ctx->insideOnly = 0;
    return ctx;
}

qivCtx *
qivCtxNix(qivCtx *ctx) {
    assert(ctx);
    /* we do not own ctx->vfl */
    /* nothing else dynamically allocated */
    free(ctx);
    return NULL;
}
