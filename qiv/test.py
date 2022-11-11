# halt if python2; thanks to https://preview.tinyurl.com/44f2beza
_x, *_y = 1, 2  # NOTE: A SyntaxError means you need python3, not python2
del _x, _y

# GLK build with: python3 build_qiv.py ~/teem-install ~/teem/python/cffi

import numpy as np
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


def check_np():
    # make matrix that indicates errors of layout
    ItoW = np.identity(3)
    ItoW[0, 1] = 0.5
    ItoW[1, 1] = 1.5
    ItoW[0, 2] = 2
    ItoW[0, 2] = 4
    ItoW[1, 2] = 3
    print(f'ItoW =\n{ItoW}')
    for order in ['C', 'F']:
        print(f'check_np ----------------- order={order}')
        dmat = np.zeros((5, 4), dtype='float64', order=order)
        dmat[0, 0] = 0.1234567890123456789
        dmat[1, 1] = 1
        dmat[2, 2] = 1
        dmat[0, -1] = 2
        dmat[-1, 0] = 3
        dmat[-1, -1] = 4
        smat = np.float32(dmat)
        print(f'smat (shape {smat.shape}) = \n{smat}')
        print('smat flags:')
        print(smat.flags)
        # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        print(f'smat (as ints) reshaped to 1D: {np.reshape(np.int32(smat), 20, order="A")}')
        print(f'smat length-{smat.shape[0]} axis has edge vector {list(ItoW[0:2, 0])}')
        print(f'smat length-{smat.shape[1]} axis has edge vector {list(ItoW[0:2, 1])}')
        print(f'dmat.dtype = {dmat.dtype}; smat.dtype = {smat.dtype}')
        # convert to qivArray and save
        q.qivArraySave(f'mat-64-{order}.txt'.encode('utf-8'), q.from_numpy(dmat, ItoW))
        q.qivArraySave(f'mat-32-{order}.txt'.encode('utf-8'), q.from_numpy(smat, ItoW))
        # both "diff mat-??-F.txt" and "diff mat-??-C.txt" should say nothing:
        # whether the precision conversion happens inside or outside qiv, result should be same
        # also, both "diff mat-32-?.txt" and "diff mat-64-?.txt" should say nothing:
        # qiv.py internal converts to C-ordering, and the saved data and meta-data should the same
        vfa = np.zeros((5, 4, 2), dtype='float32', order=order)
        vfa[0, 0, :] = [1, 2]
        vfa[-1, 0, :] = [3, 4]
        vfa[0, -1, :] = [5, 6]
        vfa[-1, -1, :] = [7, 8]
        print(f'vfa length-{vfa.shape[0]} axis has edge vector {list(ItoW[0:2, 0])}')
        print(f'vfa length-{vfa.shape[1]} axis has edge vector {list(ItoW[0:2, 1])}')
        q.qivArraySave(f'vfa-{order}.nrrd'.encode('utf-8'), q.from_numpy(vfa, ItoW))
    print('check_np ----------------- done')


check_np()


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
