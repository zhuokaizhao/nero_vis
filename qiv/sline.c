/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#include "qiv.h"
#include "qivPrivate.h"
// vvvvvv
static int
scooch(real *sv, const real pos[const 2], qivIntg intg, real hh, int sgn,
       _Bool normalize, qivArray *vfd, qivKern kern) {
    real cvec[2], k1[2], p1[2], k2[2], p2[2], k3[2], p3[2], k4[2];

#define STEP(D, POS)                                                                    \
  if (!_qivConvoEval(cvec, sgn, normalize, vfd, kern, (POS)[0], (POS)[1]))              \
    return qivStopOutside;                                                              \
  V2_SCALE(D, hh, cvec)

    if (qivIntgEuler == intg) {
        STEP(sv, pos);
    } else if (qivIntgMidpoint == intg) {
        STEP(k1, pos);
        p1[0] = pos[0] + k1[0] / 2;
        p1[1] = pos[1] + k1[1] / 2;
        STEP(sv, p1);
    } else { // RK4
        STEP(k1, pos);
        p1[0] = pos[0] + k1[0] / 2;
        p1[1] = pos[1] + k1[1] / 2;
        STEP(k2, p1);
        p2[0] = pos[0] + k2[0] / 2;
        p2[1] = pos[1] + k2[1] / 2;
        STEP(k3, p2);
        p3[0] = pos[0] + k3[0];
        p3[1] = pos[1] + k3[1];
        STEP(k4, p3);
        sv[0] = k1[0] / 6 + k2[0] / 3 + k3[0] / 3 + k4[0] / 6;
        sv[1] = k1[1] / 6 + k2[1] / 3 + k3[1] / 3 + k4[1] / 6;
    }
    if (!(isfinite(sv[0]) && isfinite(sv[1]))) {
        // got a non-finite vector along the way
        return qivStopNonfinite;
    }
    return qivStopNot;
#undef STEP
}
// ^^^^^^

static void
slineInit(qivSline *sln) {
    real nan = qivNan(0);
    V2_SET(sln->seed, nan, nan);
    sln->halfLen = sln->forwNum = sln->backNum = 0;
    sln->seedStop = sln->forwStop = sln->backStop = qivStopUnknown;
    V2_SET(sln->vecSeed, nan, nan);
    return;
}

qivSline *
qivSlineNew() {
    qivSline *sln = MALLOC(1, qivSline);
    assert(sln);
    sln->pos = NULL;
    slineInit(sln);
    return sln;
}

int
qivSlineAlloc(qivSline *sln, uint halfLen) {
    if (!sln) {
        biffAddf(QIV, "%s: got NULL pointer", __func__);
        return 1;
    }
    int doalloc;
    if (sln->pos) { // already allocated for something */
        if (sln->halfLen == halfLen) {
            // and its the right length
            doalloc = 0;
        } else {
            // else not the right length
            free(sln->pos);
            sln->pos = NULL;
            doalloc = 1;
        }
    } else {
        // sln->pos is NULL
        doalloc = 1;
    }
    if (doalloc) {
        sln->pos = MALLOC(2 * (1 + 2 * halfLen), real);
        if (!(sln->pos)) {
            biffAddf(QIV,
                     "%s: couldn't allocate pos for %u reals "
                     "(halfLen %u)",
                     __func__, 2 * (1 + 2 * halfLen), halfLen);
            return 1;
        }
    }
    slineInit(sln);
    sln->halfLen = halfLen;
    return 0;
}

qivSline *
qivSlineNix(qivSline *sln) {
    if (sln) {
        free(sln->pos);
        free(sln);
    }
    return NULL;
}

int
qivSlineTrace(_Bool doErr, qivSline *const sln,       //
              qivIntg intg, real hh, _Bool normalize, //
              qivArray *vfd, qivKern kern,            //
              real seedX, real seedY) {

    _Bool E = false; // have an error
    if (!E && !(sln && vfd)) {
        biffMaybeAddf(doErr, QIV, "%s: got NULL pointer (%p,%p)", __func__, CVOIDP(sln),
                      CVOIDP(vfd));
        E = true;
    }
    if (!E && !(sln->pos && sln->halfLen)) {
        biffMaybeAddf(doErr, "%s: need allocated sln->halfLen > 0 (not %p,%u)", __func__,
                      CVOIDP(sln->pos), sln->halfLen);
        E = true;
    }
    if (!E && 2 != vfd->channel) {
        biffMaybeAddf(doErr, QIV, "%s: need #channel == 2 (not %d)", __func__,
                      vfd->channel);
        E = true;
    }
    if (!E && qivTypeReal != vfd->type) {
        biffMaybeAddf(doErr, QIV, "%s: sorry, need array type %s (not %s)", __func__,
                      airEnumStr(qivType_ae, qivTypeReal),
                      airEnumStr(qivType_ae, vfd->type));
        E = true;
    }
    if (!E && airEnumValCheck(qivIntg_ae, intg)) {
        biffMaybeAddf(doErr, QIV, "%s: integration %d not a valid %s", __func__, intg,
                      qivIntg_ae->name);
        E = true;
    }
    if (!E && !(hh > 0)) {
        biffMaybeAddf(doErr, QIV, "%s: given step size %g not > 0", __func__, hh);
        E = true;
    }
    if (!E && !(isfinite(seedX) && isfinite(seedY))) {
        biffMaybeAddf(doErr, "%s: got non-finite (world-space) seed pos (%g,%g)",
                      __func__, seedX, seedY);
        E = true;
    }
    if (!E && airEnumValCheck(qivKern_ae, kern)) {
        biffMaybeAddf(doErr, QIV, "%s: got invalid kern %d", __func__, kern);
        E = true;
    }
    if (E) {
        /* we cannot proceed if there is an error; we have to return. What we return
        depends on doErr: if !doErr (over-ridding normal error handling), then return
        0 to (falsely) indicate there has not been an error. Else (with doErr, with
        normal error handling), return the 1 that indicates that there is a biffed error.
      */
        if (!doErr) {
            printf("%s: ERROR! but !doErr => silently retuning 0 !\n", __func__);
        }
        return doErr;
    }
    // initialize output
    real seed[2];
    V2_SET(seed, seedX, seedY);
    V2_COPY(sln->seed, seed);
    uint halfLen = sln->halfLen;
    V2_COPY(sln->pos + 2 * halfLen, seed);
    // try reconstructing at seed point
    // (all possible error checking has already been done)
    real ovec[2];
    _Bool inside = _qivConvoEval(ovec, 1 /* sgn */, false /* normalize */, vfd, kern,
                                 seed[0], seed[1]);
    real nan = qivNan(0);
    int bail = 0;
    if (!inside) {
        // seed point outside field, nowhere to go
        sln->seedStop = qivStopOutside;
        V2_SET(sln->vecSeed, nan, nan);
        bail = 1;
    } else if (normalize && !V2_LEN(ovec)) {
        // got a zero vector but want to normalize
        sln->seedStop = qivStopNonfinite;
        V2_SET(sln->vecSeed, nan, nan);
        bail = 1;
    }
    if (bail) {
        if (_qivVerbose > 20) {
            printf("%s: seed-point (%g,%g) non-starter because of %s\n", __func__,
                   seed[0], seed[1], airEnumStr(qivStop_ae, sln->seedStop));
        }
        return 0;
    }
    // else
    sln->seedStop = qivStopNot;
    V2_COPY(sln->vecSeed, ovec);
    // vvvvvv
    int dirIdx, _stop[3], *stop;
    uint stepIdx, posIdx, _snum[3], *snum;
    real dpos[2], pp[2];
    snum = _snum + 1; // can set snum[-1] and snum[1]
    stop = _stop + 1;
    for (dirIdx = -1; dirIdx <= 1; dirIdx += 2) {
        posIdx = halfLen;
        snum[dirIdx] = 0;
        V2_COPY(pp, sln->pos + 2 * posIdx);
        for (stepIdx = 0; stepIdx < halfLen; stepIdx++) {
            int retstep = scooch(dpos, pp, intg, hh, dirIdx, normalize, vfd, kern);
            if (qivStopNot != retstep) {
                stop[dirIdx] = retstep;
                break;
            }
            V2_ADD(pp, dpos, pp);
            posIdx += dirIdx;
            V2_COPY(sln->pos + 2 * posIdx, pp);
            snum[dirIdx]++;
        }
        if (stepIdx == halfLen) { // loop got to end
            stop[dirIdx] = qivStopSteps;
        }
    }
    sln->backNum = snum[-1];
    sln->forwNum = snum[1];
    sln->backStop = stop[-1];
    sln->forwStop = stop[1];
    // ^^^^^^

    return 0;
}

void
qivSlinePrint(qivSline *const sln) {
    if (!sln) {
        fprintf(stderr, "%s: got NULL pointer", __func__);
        return;
    }
    if (qivStopNot != sln->seedStop) {
        printf("went nowhere from seed %g %g ; why: %s\n", sln->seed[0], sln->seed[1],
               airEnumStr(qivStop_ae, sln->seedStop));
    } else {
        uint halfLen = sln->halfLen;
        printf("%u/%u steps backwards/forwards\n", sln->backNum, sln->forwNum);
        printf("   (why stopped backwards: %s)\n",
               airEnumStr(qivStop_ae, sln->backStop));
        /* Note that the printed posIdx indices don't always start at
           0; the index of the seedpoint is always halfLen, but the
           streamline may have been stopped from going halfLen steps
           upstream */
        uint posIdx;
        for (posIdx = halfLen - sln->backNum; posIdx <= halfLen + sln->forwNum;
             posIdx++) {
            char tmpb[512];
            if (posIdx == halfLen) {
                sprintf(tmpb, " <--seed(vec=%g,%g)", sln->vecSeed[0], sln->vecSeed[1]);
            } else if (posIdx == halfLen - sln->backNum) {
                sprintf(tmpb, " <--tail");
            } else if (posIdx == halfLen + sln->forwNum) {
                sprintf(tmpb, " <--head");
            } else {
                strcpy(tmpb, "");
            }
            printf("%u: %g %g%s\n", posIdx, (sln->pos + 2 * posIdx)[0],
                   (sln->pos + 2 * posIdx)[1], tmpb);
        }
        printf("   (why stopped forwards: %s)\n", airEnumStr(qivStop_ae, sln->forwStop));
    }
    return;
}
