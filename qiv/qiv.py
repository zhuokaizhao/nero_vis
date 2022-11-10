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
    'qivArraySet',
    'qivArrayAlloc',
    'qivArraySave',
    '*qivCtxNew',
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


# input_array is a numpy array
def set_array(input_array):

    # make sure the input array is contiguous in memory (c order)
    input_array = np.ascontiguousarray(input_array)
    if input_array.dtype == 'float32':
        nrrd_type = _teem.nrrdTypeFloat
    elif input_array.dtype == 'float64':
        nrrd_type = _teem.nrrdTypeDouble
    else:
        raise Exception(f'Unsupported data type {input_array.dtype}')

    # pointer to numpy array
    data = input_array.__array_interface__['data']

    # ItoW relationship
    edge0 = _qiv.ffi.new('double[2]')
    edge0[0] = 1.0
    edge0[1] = 0.0
    edge1 = _qiv.ffi.new('double[2]')
    edge1[0] = 0.0
    edge1[1] = 1.0
    orig = _qiv.ffi.new('double[2]')
    orig[0] = 0.0
    orig[1] = 0.0

    # initialize array
    qv = _qiv.qivArrayNew()
    _qiv.qivArraySet(
        qar=qv,
        channel=2,
        size0=input_array.shape[0],
        size1=input_array.shape[1],
        srcData=data,
        srcNType=nrrd_type,
        edge0=edge0,
        edge1=edge1,
        orig=orig,
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
    if _qiv.lib.qivRealIsDouble:
        airTypeReal = _teem.airTypeDouble
        nrrdTypeReal = _teem.nrrdTypeDouble
    else:
        airTypeReal = _teem.airTypeFloat
        nrrdTypeReal = _teem.nrrdTypeFloat
    # __all__.append('airTypeReal')
    # __all__.append('nrrdTypeReal')
    __all__.append('set_array')
