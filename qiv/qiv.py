#
# libqiv: Quick Inspection of Vector fields
# Copyright (C)  2022 University of Chicago. All rights reserved.
#
# pylint can't see inside CFFI-generated extension modules,
# and is also confused about contents of teem
# pylint: disable=c-extension-no-member,no-member

"""
This is a wrapper around "_qiv" the CFFI-generated extension
module for accessing the C shared library libqiv.{so,dylib}
The objects "exported" by this module are:
- wrappers for functions that use biff, turning biff error messages into Exceptions
- Tenums for making airEnums in libqiv useful from Python
- lib = _qiv.lib: direct references to the CFFI-generated objects
- ffi = _qiv.ffi: The value of this ffi over "from cffi import FFI; ffi = FFI()"
  is that it knows about the typedefs (especially for "real") that were cdef()'d to
  build the CFFI wrapper, which (via ffi.new) can help set up calls into libqiv.
For more about CFFI see https://cffi.readthedocs.io/en/latest/
(NOTE: GLK welcomes suggestions on how to make this more useful or pythonic)
"""

import numpy as np

try:
    import teem as _teem
except ModuleNotFoundError as _exc:
    print(
        """
***
*** qiv.py could not "import teem"
*** Maybe add the path to teem.py to environment variable PYTHONPATH ?
***
    """
    )
    raise _exc

try:
    import _qiv
except ModuleNotFoundError:
    print(
        f'*\n* {__name__}.py: failed to load libqiv extension module _qiv.\n'
        '* Did you first run "python3 build_qiv.py"?\n*\n'
    )
    raise

# halt if python2; thanks to https://preview.tinyurl.com/44f2beza
_x, *_y = 1, 2  # NOTE: A SyntaxError means you need python3, not python2
del _x, _y


def _check_risd():
    """Check that extension module _qiv and underlying libqiv agree on meaning of 'real'"""
    want_size = 8 if _qiv.lib.qivRealIsDouble else 4
    got_size = _qiv.ffi.sizeof('real')
    if got_size != want_size:
        raise RuntimeError(
            f'Extension module _qiv says sizeof(real) == {got_size}\n'
            f'(i.e. build_qiv.py was run {"WITH" if 8 == got_size else "withOUT"} "-risd")\n'
            f'but underlying libqiv shared library says sizeof(real) == {want_size}\n'
            f'(i.e. libqiv was was compiled {"WITH" if 8 == want_size else "withOUT"} '
            '"CFLAGS=-DQIV_REAL_IS_DOUBLE=1").\n'
            'Need to recompile either _qiv or libqiv for things to work.'
        )


# The functions that use biff. If function name starts with "*", a NULL return
# indicates an error, otherwise a non-zero integer return indicates an error
_BIFF_LIST = [
    'qivArrayAlloc',
    'qivArraySet',
    'qivArrayOrientationSet',
    'qivArraySyntheticFlowSet',
    'qivArraySave',
    'qivArrayBBox',
    'qivConvoEval',
    'qivSlineAlloc',
    'qivSlineTrace',
    #'qivLICEval',
]


def _biffer(func, func_name, errv, bkey):
    """Generates a biff-checking wrapper around given function (from CFFI)."""

    def wrapper(*args):
        # pass all args to underlying C function; get return value rv
        retv = func(*args)
        # we have a biff error if return value retv is error value errv
        if retv == errv:
            err = _teem.biffGetDone(bkey)
            estr = _teem.ffi.string(err).decode('ascii').rstrip()
            _teem.lib.free(err)
            raise RuntimeError(f'return value {retv} from C function "{func_name}":\n{estr}')
        return retv

    wrapper.__name__ = func_name
    wrapper.__doc__ = f"""
error-checking wrapper around C function "{func_name}":
{func.__doc__}
"""
    return wrapper


def _qivArrayNew_gc():
    """qivArrayNew wrapper adds callback to qivArrayNix upon garbage collection"""
    # without using this, the heap memory allocated within qivArrayNew is never
    # free()d, and the python runtime has no idea how to free it
    ret = _qiv.lib.qivArrayNew()
    ret = _qiv.ffi.gc(ret, _qiv.lib.qivArrayNix)
    return ret


def _qivSlineNew_gc():
    """qivSlineNew wrapper adds callback to qivSlineNix upon garbage collection"""
    # (same story as with qivArray above)
    ret = _qiv.lib.qivSlineNew()
    ret = _qiv.ffi.gc(ret, _qiv.lib.qivSlineNix)
    return ret


def _export_qiv() -> None:
    """Figures out what to export, and how, from _qiv extension module."""
    err_val = {}  # dict maps from function name to return value signifying error
    for bfunc in _BIFF_LIST:
        if '*' == bfunc[0]:
            # if function name starts with '*', then returning NULL means biff error
            fff = bfunc[1:]
            eee = ffi.NULL
        else:
            # else returning 1 indicates error
            fff = bfunc
            eee = 1
        err_val[fff] = eee
    for sym_name in dir(_qiv.lib):
        name_in = sym_name.startswith('qiv') or sym_name.startswith('QIV')
        if not name_in:
            # in the SciVis per-project libraries that this is based on, there are some
            # things like nrrdLoad that do not start with the library name, but which are
            # wanted as an export.  Not so here; user can "import teem" directly if needed
            continue
        sym = getattr(_qiv.lib, sym_name)
        # Initialize python object to export from this module for sym.
        exprt = None
        if not sym_name in err_val:
            # ... either not a function, or a function not known to use biff
            if str(sym).startswith("<cdata 'airEnum *' "):
                # sym_name is name of an airEnum, wrap it as such
                exprt = _teem.Tenum(sym, sym_name, _qiv)
            elif 'qivArrayNew' == sym_name:
                # this handled specially: use the destructor-adding wrapper
                exprt = _qivArrayNew_gc
            elif 'qivSlineNew' == sym_name:
                # this handled specially: use the destructor-adding wrapper
                exprt = _qivSlineNew_gc
            else:
                if name_in:
                    exprt = sym
                # else sym is outside qiv and is not a biff-using function,
                # so we don't make it directly available with "import qiv"
                # (though still available in qiv.lib)
        else:
            # ... or a Python wrapper around a function known to use biff.
            # The biff key is just 'qiv'
            bkey = 'qiv'.encode('ascii')
            exprt = _biffer(sym, sym_name, err_val[sym_name], bkey)
        if exprt is not None:
            globals()[sym_name] = exprt
            # Everything in C library qiv is already prefixed by "qiv" or "QIV",
            # because it is designed to play well with other C libraries (and C, just like
            # the compiler's linker, has NO notion of namespaces). So, "from qiv import *"
            # is not actually as crazy as it would be in general. We add sym_name to list
            # of what "import *" imports if it starts with qiv or QIV
            if name_in:
                __all__.append(sym_name)


def _2v(ll=None):
    """_2v(ll) returns a cdata double[2]; if ll: initialized with ll[0] and ll[1]"""
    dbp = _qiv.ffi.new('double[2]')
    if ll is not None:
        dbp[0] = ll[0]
        dbp[1] = ll[1]
    return dbp


# input_array is a numpy array
def from_numpy(np_arr, ItoW=np.identity(3)):
    """Converts from a numpy array np_arr to a qivArray. Numpy dtypes are handled as follows:
    uint8 goes to qivTypeUChar, and both float32 and float64 go to qivTypeReal (which is either
    float or double, depending on how libqiv was compiled). The passed ItoW matrix (identity by
    default) describes the numpy array orientation as follows:
    1st column ItoW[0:2, 0] == world-space position change from incrementing 1st index into np_arr,
    2nd column ItoW[0:2, 1] == world-space position change from incrementing 2nd index into np_arr,
    3rd column ItoW[0:2, 2] == world-space position of first sample (np_arr[0,0] or np_arr[0,0,:])
    """
    f_cont = np_arr.flags.f_contiguous
    c_cont = np_arr.flags.c_contiguous
    if 1 != f_cont + c_cont:
        raise RuntimeError(
            f'Expected exactly one of F,C-contiguous to be true (not {f_cont},{c_cont})'
        )
    if 2 == np_arr.ndim:
        dim = 2
    elif 3 == np_arr.ndim:
        dim = 3
    else:
        raise RuntimeError(f'Need numpy array with ndim 2 or 3 (not {np_arr.ndim})')
    # get list of sizes
    shape = list(np_arr.shape)
    if 3 == dim:
        if not (shape[0] >= 4 and shape[1] >= 4):
            raise RuntimeError(
                'first two axes (the spatial axes) should have >= 4 samples '
                f'(not {shape[0]}, {shape[1]}'
            )
        if shape[2] > 3:
            raise RuntimeError(f'last (non-spatial) axis size can be 2 or 3 (not {shape[2]})')
    if (3, 3) != ItoW.shape:
        raise RuntimeError(f'ItoW shape must be (3,3) not {ItoW.shape}')
    if [0, 0, 1] != list(ItoW[2, :]):
        raise RuntimeError(f'ItoW bottom row must be [0,0,1] not {list(ItoW[2,:])}')
    # because of using cffi.from_buffer, we do have to enforce C order
    # (else get error message "ValueError: ndarray is not C-contiguous")
    # ascontiguousarray imposes C order
    # https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html
    # NOTE: re-using the same variable np_array for what may be new memory ordering
    np_arr = np.ascontiguousarray(np_arr)
    # can handle incoming uchar, float, and double data, but inside a qivArray these
    # become only just uchar or "real", where the meaning of real (float or double)
    # is set at compile-time in libqiv
    if np_arr.dtype == 'float32':
        nrrd_type = _teem.nrrdTypeFloat
        ctype_str = 'float'
        dst_type = _qiv.lib.qivTypeReal
    elif np_arr.dtype == 'float64':
        nrrd_type = _teem.nrrdTypeDouble
        ctype_str = 'double'
        dst_type = _qiv.lib.qivTypeReal
    elif np_arr.dtype == 'uint8':
        nrrd_type = _teem.nrrdTypeUChar
        ctype_str = 'uchar'
        dst_type = _qiv.lib.qivTypeUChar
    else:
        raise RuntimeError(f'Unsupported data type {np_arr.dtype}')
    # print(f'nrrd_type = {_teem.nrrdType.str(nrrd_type)} ({nrrd_type})')
    # print(f'ctype_str = {ctype_str}')
    # (this clumsiness is why the Tenum was created, btw)
    # dtstr = _qiv.ffi.string(_qiv.lib.airEnumStr(_qiv.lib.qivType_ae, dst_type)).decode('utf-8')
    # print(f'dst_type = {dtstr} ({dst_type})')

    # "C-contiguous" (downstream of ascontiguousarray()) is slow-to-fast:
    # ---> change (in-place) to shape to match qiv's (and nrrd's) fast-to-slow
    shape.reverse()
    # form 2-vectors from columns of ItoW
    edge1 = _2v(ItoW[0:2, 0])  # evecA (first)
    edge0 = _2v(ItoW[0:2, 1])  # evecB (second)
    orig = _2v(ItoW[0:2, 2])
    # If original incoming np_arr was C-contiguous (and ascontiguousarray() kept it
    # that way) then we have a slow-to-fast ordering of syntactic axes:
    # evecA is for slower spatial axis, and edgeB is for faster, and the fast-to-slow
    # description to qiv/nrrd is edgeB, edgeA.
    # If the incoming np_array was F-contiguous (fast-to-slow), the *syntactically*
    # earlier spatial axis (described by evecA) was faster, and the later edgeB is slower.
    # But we unfortunately don't have access to original F-contiguous data; we had to
    # use ascontiguousarray(), which made it slow-to-fast, so the account above of
    # edgeB == edge0 as faster and edgeA == edge1 as slower is still correct.
    # create self-deleting qivArray pointer
    qv = _qivArrayNew_gc()
    _qiv.lib.qivArraySet(
        qv,  # qar
        1 if 2 == dim else shape[0],  # channel
        shape[dim - 2],  # size0
        shape[dim - 1],  # size1
        dst_type,  # dstType
        # https://cffi.readthedocs.io/en/latest/ref.html?highlight=from_buffer#ffi-buffer-ffi-from-buffer
        # https://stackoverflow.com/questions/16276268/how-to-pass-a-numpy-array-into-a-cffi-function-and-how-to-get-one-back-out
        _qiv.ffi.from_buffer(f'{ctype_str}*', np_arr),  # srcData
        nrrd_type,  # srcNType
        edge0,
        edge1,
        orig,
    )
    return qv


if __name__ != '__main__':  # here because of an "import"
    _check_risd()
    # The value of this ffi, as opposed to "from cffi import FFI; ffi = FFI()" is that it knows
    # about the various typedefs that were learned to build the CFFI wrapper, which may in turn
    # be useful for setting up calls into libqiv
    ffi = _qiv.ffi
    # enable access to all the un-wrapped things straight from cffi
    lib = _qiv.lib
    # initialize the list of names that will be exported with "import *"
    # (will not include ffi and lib)
    __all__ = []
    _export_qiv()
    # little hack: export something to help connect "real" type to Teem
    # (qivPrivate.h does these via #defines)
    if _qiv.lib.qivRealIsDouble:
        airTypeReal = _teem.airTypeDouble
        nrrdTypeReal = _teem.nrrdTypeDouble
    else:
        airTypeReal = _teem.airTypeFloat
        nrrdTypeReal = _teem.nrrdTypeFloat
    # __all__.append('airTypeReal')
    # __all__.append('nrrdTypeReal')
    __all__.append('from_numpy')
