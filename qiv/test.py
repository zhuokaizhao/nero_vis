# halt if python2; thanks to https://preview.tinyurl.com/44f2beza
_x, *_y = 1, 2  # NOTE: A SyntaxError means you need python3, not python2
del _x, _y

# GLK build with: python3 build_qiv.py ~/teem-install ~/teem/python/cffi

import teem as tm
from qiv import *
import _qiv


def check_enums():
    ers = _qiv.ffi.new(f'char[{tm.AIR_STRLEN_LARGE}]')
    problem = False
    for enm in [qivType_ae, qivIntg_ae, qivStop_ae, qivKern_ae]:
        if _qiv.lib.airEnumCheck(ers, enm()):
            print(f'problem with airEnum {enm.name}:')
            print(_qiv.ffi.string(ers).decode('utf-8'))
            problem = True
    # buffer pointed to by ers is free'd when ers is garbage-collected
    if problem:
        exit()
    print('enums all ok')


check_enums()

nv = tm.nrrdNew()
# rvectr sdg -sz 30 50 -r 20 -s 0.2 -p 0 0 1 0 0 1 -o ident.nrrd
tm.nrrdLoad(nv, b'noise.nrrd', tm.NULL)

# rvectr sdg -sz 10 8 -l 10 8 -r 0 -s 0 -p 0 0 1 0 0 1 -c 4.5 3.5 -o 000.nrrd
# tm.nrrdLoad(nv, b'000.nrrd', tm.NULL)

qa = qivArrayNew()
qivArraySet(
    qa,
    nv.axis[0].size,
    nv.axis[1].size,
    nv.axis[2].size,
    qivTypeReal,
    nv.data,
    nv.type,
    nv.axis[1].spaceDirection,
    nv.axis[2].spaceDirection,
    nv.spaceOrigin,
)

# qivVerboseSet(1)

qtent = qivCtxNew(qa, qivKernTent)
qctmr = qivCtxNew(qa, qivKernCtmr)
qbspl = qivCtxNew(qa, qivKernBspln)
tp = [[0.0, 0.0], [0.10, 0.33], [0.11, 0.32], [0.12, 0.31], [0.13, 0.30], [-0.2222, 0.5555]]
if True:
    for p in tp:
        qivConvoEval(qtent, p[0], p[1], 1, 0)
        print(f'(v*tent)({p[0]},{p[1]}) = ({qtent.vec[0]},{qtent.vec[1]})')
        qivConvoEval(qctmr, p[0], p[1], 1, 0)
        print(f'(v*ctmr)({p[0]},{p[1]}) = ({qctmr.vec[0]},{qctmr.vec[1]})')
        qivConvoEval(qbspl, p[0], p[1], 1, 0)
        print(f'(v*bspl)({p[0]},{p[1]}) = ({qbspl.vec[0]},{qbspl.vec[1]})')

sl = qivSlineNew()
qivSlineTrace(sl, 0.6, 0.8, 10, 0.1, 0, qivIntgRK4, qtent)
qivSlinePrint(sl)
