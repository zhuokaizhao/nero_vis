# halt if python2; thanks to https://preview.tinyurl.com/44f2beza
_x, *_y = 1, 2  # NOTE: A SyntaxError means you need python3, not python2
del _x, _y

# GLK build with: python3 build_qiv.py ~/teem-install ~/teem/python/cffi

import teem as tm
import qiv as q


def check_enums():
    ers = q.ffi.new(f'char[{tm.AIR_STRLEN_LARGE}]')
    problem = False
    for enm in [q.qivType_ae, q.qivIntg_ae, q.qivStop_ae, q.qivKern_ae]:
        if q.lib.airEnumCheck(ers, enm()):
            print(f'problem with airEnum {enm.name}:')
            print(q.ffi.string(ers).decode('utf-8'))
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

qa = q.qivArrayNew()
q.qivArraySet(
    qa,
    nv.axis[0].size,
    nv.axis[1].size,
    nv.axis[2].size,
    q.qivTypeReal,
    nv.data,
    nv.type,
    nv.axis[1].spaceDirection,
    nv.axis[2].spaceDirection,
    nv.spaceOrigin,
)
q.qivArraySave(b'tmp.nrrd', qa)

# qivVerboseSet(1)

tp = [[0.0, 0.0], [0.11, 0.32], [0.12, 0.31], [-0.2222, 0.5555], [-222, 555]]
inside = q.ffi.new('_Bool[1]')
kern = {'tent': q.qivKernTent, 'ctmr': q.qivKernCtmr, 'bspln': q.qivKernBspln}
ovec = q.ffi.new('real[2]')
if True:
    for kn, kp in kern.items():
        print(f'with {kn} kernel')
        for p in tp:
            q.qivConvoEval(inside, ovec, 1, False, qa, kp, p[0], p[1])
            print(f'(v⊛{kn})({p[0]},{p[1]}) = ({ovec[0]},{ovec[1]})  inside={inside[0]}')

sl = q.qivSlineNew()
q.qivSlineAlloc(sl, 10)
q.qivSlineTrace(
    sl,  # sln
    q.qivIntgRK4,  # intg
    0.1,  # hh
    False,  # normalize
    qa,  # vfd
    q.qivKernTent,  # kern
    0.6,  # seedX
    0.8,  # seedY
)
q.qivSlinePrint(sl)
# compare to: rvectr sline -i noise.nrrd -k tent -s 0.6 0.8 -h 0.1 -l 10 -intg rk4
