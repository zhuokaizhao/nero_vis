/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#include "qiv.h"
#include "qivPrivate.h"

void
qivConvoEval(qivCtx *ctx, real xw, real yw) {
    if (!ctx) {
        fprintf(stderr, "%s: got NULL pointer!", __func__);
        return;
    }
    real ip[3], wp[3] = {xw, yw, 1};
    MV3_MUL(ip, ctx->WtoI, wp);

    printf("%s: world (%g,%g) --> index (%g,%g)\n", __func__, xw, yw, ip[0], ip[1]);
}
