/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#include "qiv.h"
#include "qivPrivate.h"

/* copied from SciVis kernel.c */
static real
Bspln3Eval(real x) {
    real ret;
    x = fabs(x);
    if (x < 1) {
        ret = (real)2 / 3 + x * x * (-1 + x / 2);
    } else if (x < 2) {
        x -= 1;
        ret = (real)1 / 6 + x * (-(real)1 / 2 + x * ((real)1 / 2 - x / 6));
    } else {
        ret = 0;
    }
    return ret;
}

/* does all the error checking that one can do for the arguments to qivLICEval */
int
qivLICEvalCheck(const qivArray *rnd, _Bool rndLinterp,  //
                qivSline *const sln,                    //
                qivIntg intg, real hh, _Bool normalize, //
                qivArray *vfd, qivKern kern,            //
                real xw, real yw) {
    if (!rnd) {
        biffAddf("%s: got NULL rnd", __func__);
        return 1;
    }
    (void)(rndLinterp); // sshshshs unused-parameter warning
    (void)(normalize);  // sshshshs unused-parameter warning
    if (!(1 == rnd->channel && qivTypeReal == rnd->type)) {
        biffAddf("%s: need rnd as 1-channel array of real (not %u-channel of %s)",
                 __func__, rnd->channel, airEnumStr(qivType_ae, rnd->type));
        return 1;
    }
    if (!sln) {
        biffAddf("%s: got NULL sln", __func__);
        return 1;
    }
    if (!(sln->pos && sln->halfLen)) {
        biffAddf("%s: need allocated sln->halfLen > 0 (not %p,%u)", __func__,
                 CVOIDP(sln->pos), sln->halfLen);
        return 1;
    }
    if (airEnumValCheck(qivIntg_ae, intg)) {
        biffAddf("%s: got invalid %s %d", __func__, qivIntg_ae->name, intg);
        return 1;
    }
    if (!(hh > 0)) {
        biffAddf("%s: need hh > 0 (not %g)", __func__, hh);
        return 1;
    }
    if (!vfd) {
        biffAddf("%s: got NULL vfd", __func__);
        return 1;
    }
    if (!(2 == vfd->channel && qivTypeReal == vfd->type)) {
        biffAddf("%s: need vfd as 2-channel array of real (not %u-channel of %s)",
                 __func__, vfd->channel, airEnumStr(qivType_ae, vfd->type));
        return 1;
    }
    if (airEnumValCheck(qivKern_ae, kern)) {
        biffAddf("%s: got invalid %s %d", __func__, qivKern_ae->name, kern);
        return 1;
    }
    if (!(isfinite(xw) && isfinite(yw))) {
        biffAddf("%s: got non-finite (world-space) seed pos (%g,%g)", __func__, xw, yw);
        return 1;
    }
    return 0;
}

/* returns the LIC evaluation at (xw,yw) for the given parameters.
   Assumes that the caller has checked all the parameters with qivLICEvalCheck() */
real
qivLICEval(const qivArray *rnd, _Bool rndLinterp,  //
           qivSline *const sln,                    //
           qivIntg intg, real hh, _Bool normalize, //
           qivArray *vfd, qivKern kern,            //
           real xw, real yw) {

    qivSlineTrace(true /* noErr */, sln, intg, hh, normalize, vfd, kern, xw, yw);
    if (qivStopNot != sln->seedStop) {
        /* if the streamline could not get past the seedpoint, or couldn't
           even probe the vector field at the seedpoint, then there's no point
           in trying to do convolution of noise along the streamline */
        return 0;
    }
    // vvvvvv
    uint size0 = rnd->size0;
    uint size1 = rnd->size1;
    uint halfLen = sln->halfLen;
    if (_qivVerbose > 10) {
        printf("%s: bspln3 on streamline indices halfLen-backNum=%u-%u=%u to "
               "halfLen+forwNum=%u+%u=%u\n",
               __func__, halfLen, sln->backNum, halfLen - sln->backNum, halfLen,
               sln->forwNum, halfLen + sln->forwNum);
    }
    real spos[3] = {0, 0, 1};
    real ripos[3];
    real tsum = 0, wsum = 0;
    for (uint posIdx = halfLen - sln->backNum; //
         posIdx <= halfLen + sln->forwNum;     //
         posIdx++) {
        real ff = (posIdx + 0.5) / (2 * halfLen + 1);
        real kp = (1 - ff) * (-2) + ff * 2;
        real ww = Bspln3Eval(kp);
        V2_COPY(spos, sln->pos + 2 * posIdx);
        MV3_MUL(ripos, rnd->WtoI, spos);
        if (_qivVerbose > 10) {
            printf(
              "%s: streamline index %u -> %g within bspln3 support [-2,2] -> ww=%g\n",
              __func__, posIdx, kp, ww);
            printf("       streamline wpos %g %g -> rnd ipos %g %g\n", spos[0], spos[1],
                   ripos[0], ripos[1]);
        }
        if (rndLinterp) {
            int ri = (int)floor(ripos[0]);
            int rj = (int)floor(ripos[1]);
            real a0 = ripos[0] - ri;
            real a1 = ripos[1] - rj;
            if ((uint)rj == size1 - 1) {
                rj--;
                a1++;
            }
            if ((uint)ri == size0 - 1) {
                ri--;
                a0++;
            }
            if (_qivVerbose > 10)
                printf("        w/ rndLinterp: ipos %g %g = %d+%g,%d+%g\n", ripos[0],
                       ripos[1], ri, a0, rj, a1);
            if (0 <= ri && (uint)ri < size0 - 1 && a0 <= 1 && 0 <= rj
                && (uint)rj < size1 - 1 && a1 <= 1) {
                const real *rr = rnd->data.rl + ri + size0 * rj;
                real rlrp = ((1 - a0) * (1 - a1) * rr[0] + a0 * (1 - a1) * rr[1]
                             + (1 - a0) * a1 * rr[size0] + a0 * a1 * rr[size0 + 1]);
                tsum += ww * rlrp;
                wsum += ww * ww;
                if (_qivVerbose > 10) {
                    printf(
                      "        w/ rndLinterp: %d+%g,%d+%g inside -> rnd (base index %u) "
                      "\n"
                      "        rnd values (%g,%g,%g,%g) weighted by (%g,%g,%g,%g)"
                      "=%g=rlrp; ww*rlrp = %g*%g = %g\n"
                      "        tsum+=%g --> %g;  wsum+=|ww| --> %g)\n",
                      ri, a0, rj, a1, ri + size0 * rj, rr[0], rr[1], rr[size0],
                      rr[size0 + 1], (1 - a0) * (1 - a1), a0 * (1 - a1), (1 - a0) * a1,
                      a0 * a1, rlrp, ww, rlrp, ww * rlrp, ww * rlrp, tsum, wsum);
                }
            } else {
                if (_qivVerbose > 10) {
                    printf("        w/ rndLinterp: %d+%g,%d+%g outside; not changing "
                           "tsum=%g, wsum=%g\n",
                           ri, a0, rj, a1, tsum, wsum);
                }
            }
        } else {
            int ri = (int)floor((real)0.5 + ripos[0]);
            int rj = (int)floor((real)0.5 + ripos[1]);
            if (0 <= ri && (uint)ri < size0 && 0 <= rj && (uint)rj < size1) {
                tsum += ww * rnd->data.rl[ri + size0 * rj];
                wsum += ww * ww;
                if (_qivVerbose > 10) {
                    printf("        (nearest-neighbor rnd interp) %d,%d inside --> "
                           "wsum += |ww| --> %g\n"
                           "        tsum += ww*rnd[%u] = %g*%g --> %g\n",
                           ri, rj, wsum, ri + size0 * rj, ww,
                           rnd->data.rl[ri + size0 * rj], tsum);
                }
            }
        }
    }
    real result;
    if (wsum) {
        result = tsum / wsum;
        if (_qivVerbose > 10)
            printf("%s: result = tsum/wsum = %g/%g =%g\n", __func__, tsum, wsum, result);
    } else {
        result = 0;
        if (_qivVerbose > 10) printf("%s: wsum == 0 --> result=0\n", __func__);
    }
    // ^^^^^^
    return result;
}
