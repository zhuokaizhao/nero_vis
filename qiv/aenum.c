/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#include "qiv.h"
#include "qivPrivate.h"

static const airEnum _qivIntg_ae
  = {"integration method",
     3,
     (const char *[]){"(unknown_intg)", "euler", "midpoint", "rk4"},
     (int[]){qivIntgUnknown, qivIntgEuler, qivIntgMidpoint, qivIntgRK4},
     (const char *[]){
       "unknown integration",
       "Euler integration",
       "Midpoint-method integration (RK2)",
       "Runge-Kutta 4th order (RK4)",
     },
     (const char *[]){"euler", "midpoint", "rk2", "rk4", ""},
     (int[]){qivIntgEuler, qivIntgMidpoint, qivIntgMidpoint, qivIntgRK4},
     AIR_FALSE};
const airEnum *const qivIntg_ae = &_qivIntg_ae;

static const airEnum _qivStop_ae
  = {"why integration stopped (or never started)",
     4,
     (const char *[]){"(unknown_stop)", "not", "outside", "nonfinite", "steps"},
     (int[]){qivStopUnknown, qivStopNot, qivStopOutside, qivStopNonfinite, qivStopSteps},
     (const char *[]){
       "unknown stop",
       "actually, not stopped",
       "stopped outside field domain",
       "stopped after non-finite vector field value",
       "stopped after taking max number of steps",
     },
     (const char *[]){"not", "outside", "out", "nonfinite", "steps", ""},
     (int[]){qivStopNot, qivStopOutside, qivStopOutside, qivStopNonfinite, qivStopSteps},
     AIR_FALSE};
const airEnum *const qivStop_ae = &_qivStop_ae;

static const airEnum _qivKern_ae
  = {"the different reconstruction kernels we know about",
     4,
     (const char *[]){"(unknown_kern)", "box", "tent", "ctmr", "bspln"},
     (int[]){qivKernUnknown, qivKernBox, qivKernTent, qivKernCtmr, qivKernBspln},
     (const char *[]){
       "unknown kernel",
       "box == nearest neighbor",
       "tent == bilinear",
       "Catmull-Rom",
       "uniform cubic B-spline",
     },
     NULL,
     NULL,
     AIR_FALSE};
const airEnum *const qivKern_ae = &_qivKern_ae;
