# halt if python2; thanks to https://preview.tinyurl.com/44f2beza
_x, *_y = 1, 2  # NOTE: A SyntaxError means you need python3, not python2
del _x, _y

import qiv as q


def v2p(ll=None):
    """v2p(ll) returns a cdata double[2]; if ll: initialized with ll[0] and ll[1]"""
    db = q.ffi.new('double[2]')
    if ll:
        db[0] = ll[0]
        db[1] = ll[1]
    return db


qa = q.qivArrayNew()
sz0 = 40
sz1 = 30
ssp = 0.5   # sample spacing (isotropic)
q.qivArrayAlloc(qa, 2, sz0, sz1, q.qivTypeReal)
q.qivArrayOrientationSet(
    qa,
    v2p([ssp, 0]),  # edge0
    v2p([0, ssp]),  # edge1
    # orig = location of first sample, with grid centered on origin
    v2p([-float(sz0 - 1) * ssp / 2, -float(sz1 - 1) * ssp / 2]),
)
# this samples a flow that rotates the first world-space basis vector towards the second
q.qivArraySyntheticFlowSet(qa, 0, 0, 0, -1, 1, 0)
xyMin = v2p()
xyMax = v2p()
q.qivArrayBBox(xyMin, xyMax, qa)
print(f'array bbox min={xyMin[0]},{xyMin[1]}  max={xyMax[0]},{xyMax[1]}')
# saving for reference
q.qivArraySave(b'flow.nrrd', qa)

sl = q.qivSlineNew()
q.qivSlineAlloc(sl, 10)
q.qivSlineTrace(
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
