#
# libqiv: Quick Inspection of Vector fields
# Copyright (C)  2022 University of Chicago. All rights reserved.
"""
Builds the _qiv CFFI python extension module (in file named _qiv.<platform>.so,
where <platform> is something about your Python version and OS), which links into the
shared libqiv library containing the project C code.
"""
import os
import sys
import argparse
import pathlib
import cffi

# TODO: import teem after appending to sys.path as needed

# halt if python2; thanks to https://preview.tinyurl.com/44f2beza
_x, *_y = 1, 2  # NOTE: A SyntaxError means you need python3, not python2
del _x, _y


def drop_at(bye: str, num: int, lines: list[str]) -> int:
    """From list of strings lines, drop num lines starting with first occurance of bye"""
    # idx = lines.index(bye)
    try:
        idx = lines.index(bye)
    except ValueError as exc:
        # so that exact line wasn't present, but maybe something starting with that?
        idx = -1
        for ii, line in enumerate(lines):
            if VERBOSE > 1:
                print(f'does {ii}:|{line}| start with |{bye}|')
            if line.startswith(bye):
                idx = ii
                break
        if -1 == idx:
            # no line started with bye; error
            raise exc
    for ii in range(num):
        if VERBOSE > 1:
            print(f'dropping |{lines[idx]}| ({ii}/{num} after "{bye}")')
        lines.pop(idx)
    return idx


def header(hdr_fname: str, risd: bool):
    with open(hdr_fname, 'r', encoding='utf-8') as file:
        lines = [line.rstrip() for line in file.readlines()]
    # lose the directives
    drop_at('#ifndef QIV_HAS_BEEN_INCLUDED', 2, lines)
    idx = drop_at('// begin includes', 11, lines)
    if '// end includes' != lines[idx]:
        raise RuntimeError(
            f'# of lines between "// begin includes" and "// end includes" not expected'
        )
    drop_at('// end includes', 1, lines)
    drop_at('#ifdef __cplusplus', 3, lines)   # first, at top of file
    drop_at('#ifdef __cplusplus', 3, lines)   # second, at bottom of file
    drop_at('#endif // QIV_HAS_BEEN_INCLUDED', 1, lines)
    # handle the real typedef
    idx = drop_at('#ifndef QIV_REAL_IS_DOUBLE', 8, lines)
    lines.insert(idx, f"typedef {'double' if risd else 'float'} real;")
    # the QIV biff key isn't needed
    drop_at('#define QIV qivBiffKey', 1, lines)
    if VERBOSE > 2:
        print(f'pre-processed {hdr_fname} follows:')
        for lix, line in enumerate(lines):
            print(f'{lix+1}: {line}')
    return lines


def check_opts(teem_install: str, teem_python: str) -> None:
    """Error-checking on command-line options"""
    if not os.path.isdir(teem_install):
        raise RuntimeError(f'Teem install path {teem_install} not a directory')
    if not os.path.isdir(teem_install + '/lib'):
        raise RuntimeError(f'Teem install path {teem_install} missing "lib" subdirectory')
    if not os.path.isdir(teem_install + '/include'):
        raise RuntimeError(f'Teem install path {teem_install} missing "include" subdirectory')
    libteem = teem_install + f'/lib/libteem.{SHEXT}'
    if not os.path.isfile(libteem):
        raise RuntimeError(f'Teem expected shared library {libteem} not a file')
    if not os.path.isdir(teem_python):
        raise RuntimeError(f'Teem python/cffi path {teem_python} not a directory')
    if not os.path.isfile(teem_python + '/cdef_teem.h'):
        raise RuntimeError(
            f'Teem python/cffi path {teem_python} missing cdef_teem.h header; '
            'did you run build_teem.py there first?'
        )


def build(teem_install: str, teem_python: str, risd: bool, use_int: bool) -> None:
    """
    Sets up and makes calls into cffi.FFI() to compile Python _qiv extension module
    that links into libqiv shared library
    """
    # given reliance on files in specific places; change to dir containing this file
    os.chdir(pathlib.Path(os.path.realpath(__file__)).parent)
    here = os.getcwd()
    # bail if libqiv shared library not here
    if not os.path.isfile(f'libqiv.{SHEXT}'):
        raise RuntimeError(f'Not seeing libqiv.{SHEXT} in {here}; make sure to run "make" first!')
    shlib_path = teem_install + '/lib'
    source_args = {
        'libraries': ['qiv', 'teem'],  # but apparently not 'png' and 'z' ?
        'include_dirs': ['.', teem_install + '/include'],
        'library_dirs': [here, shlib_path],
        # The next arg teaches the extension library about the paths that the dynamic linker
        # should look in for other libraries we depend on.
        # We are avoiding any reliace on environment variables like
        # LD_LIBRARY_PATH on linux or DYLD_LIBRARY_PATH on Mac (on recent Macs the System
        # Integrity Protection (SIP) actually disables DYLD_LIBRARY_PATH).
        # On linux, paths listed here are passed to -Wl,--enable-new-dtags,-R<dir>
        # and "readelf -d __lib_.cpython*-linux-gnu.so | grep PATH" should show these paths,
        # and "ldd __lib_.cpython*-linux-gnu.so" should show where dependencies were found.
        # On Mac, paths listed should be passed to -Wl,-rpath,<dir>, and you can see those
        # with "otool -l __lib_.cpython*-darwin.so", in the LC_RPATH sections. However, in
        # at least one case GLK observed, this didn't happen, so we redundantly also set
        # rpath directly for Macs, in the next statement below.
        # (Also for Mac: note that the "-int" option sets the use_int argument above, which
        # runs the "install_name_tool" utility below, but this should not be necessary if
        # the rpath has been set correctly)
        'runtime_library_dirs': [here, shlib_path],
        # https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension
        'undef_macros': ['NDEBUG'],  # keep asserts() and _lib_Verbose as normal
    }
    if risd:
        source_args['extra_compile_args'] = ['-DQIV_REAL_IS_DOUBLE=1']
    if sys.platform == 'darwin':  # make extra sure that rpath is set on Mac
        source_args['extra_link_args'] = [f'-Wl,-rpath,{P}' for P in [here, shlib_path]]
    ffibld = cffi.FFI()
    # declare this so that qiv.py can call free() on biff messages
    ffibld.cdef('extern void free(void *);')
    # We really want to be able to use "ffibld.include(teem.ffi)" after "import teem"
    # https://cffi.readthedocs.io/en/latest/cdef.html?highlight=ffi.include#ffi-ffibuilder-include-combining-multiple-cffi-interfaces
    # however this led to errors about cffi.FFI versus FFI, and that is apparently
    # the FFIs involved should from the *builders*, not the FFIs available in .lib.ffi
    # of the imported extension module.
    # Super-dumb hack for now: reduplicate ALL of Teem's low-level python bindings here
    with open(f'{teem_python}/cdef_teem.h', 'r', encoding='utf-8') as file:
        ffibld.cdef(file.read())
    # now pre-process and then declare contents of qiv.h
    ffibld.cdef('\n'.join(header('qiv.h', risd)))
    if VERBOSE:
        print('## build_qiv.py: calling set_source with ...')
        for key, val in source_args.items():
            print(f'   {key} = {val}')
    ffibld.set_source(
        f'_qiv',
        f"""
#include "{teem_python}/cdef_teem.h"
#include "qiv.h"
""",
        **source_args,
    )
    if VERBOSE:
        print(f'## build_qiv.py: compiling _qiv (SLOW because of hack) ...')
    sopath = ffibld.compile(verbose=(VERBOSE > 0))
    somade = os.path.basename(sopath)
    if VERBOSE:
        print(f'## build_qiv.py: ... done compiling _qiv')
        print(f'## build_qiv.py: created extension library: {somade}')
    if use_int:  # should only be true on mac
        cmd = (
            'install_name_tool -change @rpath/libteem.dylib '
            f'{teem_install}/lib/libteem.dylib {somade}'
        )
        if VERBOSE:
            print('## build_qiv.py: setting full path to libteem.dylib with:')
            print('   ', cmd)
        if os.system(cmd):
            raise RuntimeError(f'due to trying to set full path to libteem.dylib in {somade}')


def parse_args():
    """
    Set up and run argparse command-line parser
    """
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(
        description='Utility for compiling CFFI-based '
        'python3 extension around libqiv shared library'
    )
    parser.add_argument(
        '-v',
        metavar='verbosity',
        type=int,
        default=1,
        required=False,
        help='verbosity level (use "-v 0" for silent operation)',
    )
    parser.add_argument(
        '-risd',
        action='store_true',
        default=False,
        help=(
            'By default the "real" type within qiv is typedef\'d to float. '
            'If qiv has been compiled with "make CFLAGS=-DQIV_REAL_IS_DOUBLE=1", '
            'then the -risd option should be used here.'
        ),
    )
    if sys.platform == 'darwin':  # mac
        parser.add_argument(
            '-int',
            action='store_true',
            default=False,
            help=(
                'after creating cffi extension library, store in it the '
                'explicit teem_path/lib path to libteem.dylib, using '
                'install_name_tool.'
            ),
        )
    parser.add_argument(
        'teem_install',
        type=str,
        help=(
            'path into which CMake has built and installed Teem '
            '(should have "include" and "lib" subdirectories)'
        ),
    )
    parser.add_argument(
        'teem_python',
        type=str,
        help=(
            'path to teem/python/cffi directory where Teem extension module has beem built; '
            'a side-effect of which is creating a "cdef_teem.h" header file, which '
            '(as a terrible hack!) is going to be read here.'
        ),
    )
    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    VERBOSE = ARGS.v
    if sys.platform == 'darwin':  # mac
        SHEXT = 'dylib'
    else:
        SHEXT = 'so'
    check_opts(ARGS.teem_install, ARGS.teem_python)
    build(
        ARGS.teem_install,
        ARGS.teem_python,
        ARGS.risd,
        ARGS.int if sys.platform == 'darwin' else False,
    )

# https://stackoverflow.com/questions/45766740/typeerror-when-using-cffi-to-test-c-code-using-struct
# https://stackoverflow.com/questions/68894497/python3-cffi-is-there-a-way-to-merge-two-independent-ffis-into-one
# https://stackoverflow.com/questions/54986536/pass-objects-between-libraries-in-python-cffi
# https://cffi.readthedocs.io/en/latest/cdef.html?highlight=ffi.include#ffi-ffibuilder-include-combining-multiple-cffi-interfaces

# https://cffi.readthedocs.io/en/latest/cdef.html
