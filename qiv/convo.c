/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#include "qiv.h"
#include "qivPrivate.h"

/* clang-format off */
// (copied from SciVis kernel.c)
static void CtmrApply(real *ww, real xa) {
    ww[0] = -((xa-1)*(xa-1)*xa)/2;
    ww[1] = (2 + xa*xa*(-5 + 3*xa))/2;
    ww[2] = (xa + xa*xa*(4 - 3*xa))/2;
    ww[3] = ((-1 + xa)*xa*xa)/2;
}
static void Bspln3Apply(real *ww, real xa) {
    ww[0] = (real)1/6 + xa*(-(real)1/2 + xa*((real)1/2 - xa/6));
    ww[1] = (real)2/3 + xa*xa*(-1 + xa/2);
    ww[2] = (real)1/6 + xa*((real)1/2 + xa*((real)1/2 - xa/2));
    ww[3] = xa*xa*xa/6;
}
/* clang-format on */

void
qivConvoEval(qivCtx *ctx, real xw, real yw, int sgn, int norm) {
    if (!ctx) {
        fprintf(stderr, "%s: got NULL pointer!", __func__);
        return;
    }
    real xi, yi;
    {
        const real *m = ctx->WtoI;
        xi = m[0] * xw + m[1] * yw + m[2];
        yi = m[3] * xw + m[4] * yw + m[5];
    }
    if (_qivVerbose > 2) {
        printf("%s: world (%g,%g) --> index (%g,%g)\n", __func__, xw, yw, xi, yi);
    }
    int _ii, ii, _jj, jj, sz0 = (int)ctx->qar->size0, sz1 = (int)ctx->qar->size1;
    real aa, bb, uu[4], vv[4], tt[4];
    // safe because only real-type arrays can go into a ctx
    const real *vd = ctx->qar->data.rl, *v0, *v1, *v2, *v3;
    real *ovec = ctx->vec;
    switch (ctx->kern) {
    case qivKernBox:
        _ii = (int)round(xi);
        ii = AIR_CLAMP(0, _ii, sz0 - 1);
        _jj = (int)round(yi);
        jj = AIR_CLAMP(0, _jj, sz1 - 1);
        V2_COPY(ovec, vd + 2 * (ii + sz0 * jj));
        break;
    case qivKernTent:
        _ii = (int)floor(xi);
        aa = xi - _ii;
        ii = AIR_CLAMP(0, _ii, sz0 - 2);
        _jj = (int)floor(yi);
        bb = yi - _jj;
        jj = AIR_CLAMP(0, _jj, sz1 - 2);
        v0 = vd + 2 * (ii + sz0 * jj);
        v1 = v0 + 2 * sz0;
        /* clang-format off */
#define DO(I) \
        ovec[I] =  (1-aa)*(1-bb)*(v0+0)[I] + aa*(1-bb)*(v0+2)[I] + \
                   (1-aa)*  bb  *(v1+0)[I] + aa*  bb  *(v1+2)[I]
        DO(0);
        DO(1);
#undef DO
        /* clang-format on */
        break;
    case qivKernCtmr:
    case qivKernBspln:
        _ii = (int)floor(xi);
        aa = xi - _ii;
        ii = AIR_CLAMP(1, _ii, sz0 - 3);
        _jj = (int)floor(yi);
        bb = yi - _jj;
        jj = AIR_CLAMP(1, _jj, sz1 - 3);
        if (_qivVerbose > 3) {
            printf("%s: (%g,%g) -> _(%d,%d) ; (%d,%d) + (%g,%g)\n", __func__, xi, yi,
                   _ii, _jj, ii, jj, aa, bb);
        }
        if (qivKernCtmr == ctx->kern) {
            CtmrApply(uu, aa);
            CtmrApply(vv, bb);
        } else {
            Bspln3Apply(uu, aa);
            Bspln3Apply(vv, bb);
        }
        v0 = vd + 2 * (ii - 1 + sz0 * (jj - 1)); // have to decrement along both axes
        v1 = v0 + 2 * sz0;
        v2 = v1 + 2 * sz0;
        v3 = v1 + 4 * sz0;
        /* clang-format off */
#define DO(I) \
        tt[0] = uu[0]*(v0+0)[I] + uu[1]*(v0+2)[I] + uu[2]*(v0+4)[I] + uu[3]*(v0+6)[I]; \
        tt[1] = uu[0]*(v1+0)[I] + uu[1]*(v1+2)[I] + uu[2]*(v1+4)[I] + uu[3]*(v1+6)[I]; \
        tt[2] = uu[0]*(v2+0)[I] + uu[1]*(v2+2)[I] + uu[2]*(v2+4)[I] + uu[3]*(v2+6)[I]; \
        tt[3] = uu[0]*(v3+0)[I] + uu[1]*(v3+2)[I] + uu[2]*(v3+4)[I] + uu[3]*(v3+6)[I]; \
        ovec[I] = ELL_4V_DOT(tt, vv)
        DO(0);
        DO(1);
#undef DO
        /* clang-format on */
        break;
    default:
        printf("%s: sorry kernel %s (%d) not implemented\n", __func__,
               airEnumStr(qivKern_ae, ctx->kern), ctx->kern);
        _ii = 0; // bogus values so that ctx->inside is set to 0
        ii = 1;
        _jj = 0;
        jj = 1;
        break;
    }
    ctx->inside = (ii == _ii && jj == _jj);
    if (sgn) {
        ovec[0] *= sgn;
        ovec[1] *= sgn;
    }
    if (norm) {
        real len = ELL_2V_DOT(ovec, ovec);
        if (len) {
            len = sqrt(len);
            V2_SCALE(ovec, 1 / len, ovec);
        }
    }
    return;
}
