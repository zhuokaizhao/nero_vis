# halt if python2; thanks to https://preview.tinyurl.com/44f2beza
_x, *_y = 1, 2  # NOTE: A SyntaxError means you need python3, not python2
del _x, _y

import qiv as q


qa = q.qivArrayNew()
sz0 = 40
sz1 = 30
ssp = 0.5   # sample spacing (isotropic)
q.qivArrayAlloc(qa, 2, sz0, sz1, q.qivTypeReal)
q.qivArrayOrientationSet(
    qa,
    q.dbl2([ssp, 0]),  # edge0
    q.dbl2([0, ssp]),  # edge1
    # orig = location of first sample, with grid centered on origin
    q.dbl2([-float(sz0 - 1) * ssp / 2, -float(sz1 - 1) * ssp / 2]),
)
# this samples a flow that rotates the first world-space basis vector towards the second
q.qivArraySyntheticFlowSet(qa, 0, 0, 0, -1, 1, 0)
# saving for reference
q.qivArraySave(b'flow.nrrd', qa)

sl = q.qivSlineNew()
q.qivSlineAlloc(sl, 10)
q.qivSlineTrace(
    True,  # doErr
    sl,  # sln
    q.qivIntgMidpoint,  # intg
    0.25,  # hh
    False,  # normalize
    qa,  # vfd
    q.qivKernTent,  # kern
    5.0,  # seedX
    0.0,  # seedY
)
q.qivSlinePrint(sl)
# compare to: rvectr sline -i flow.nrrd -k tent -s 5 0 -h 0.25 -l 10 -intg midpoint
