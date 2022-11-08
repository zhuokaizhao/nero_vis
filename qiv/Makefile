##
##  libqiv: Quick Inspection of Vector fields
##  Copyright (C)  2022 University of Chicago. All rights reserved.
##


## (much of this is copied from code of GLK's CMSC 23710 SciVis class)

# we try to tell Make to skip its mostly outdated "implicit rules", and we
# simulate calling make with "-r" which turns them off
.SUFFIXES:
ifeq ($(filter -r,$(MAKEFLAGS)),) # goo.gl/xMRkAW
MAKEFLAGS += -r
endif
# these targets are not actually files
.PHONY: clean

# These variables identify (in way consistent across projects) what to build
LIB := qiv

# With shared (dynamically linked) libraries, things depend more on OS
# than with static; name of library ends with .dylib or .so
ifeq ($(shell uname -s),Darwin)
  SHEXT := dylib
else
  SHEXT := so
endif
# set name of shared library to create
LLIBSH := libqiv.$(SHEXT)
ifeq ($(shell uname -s),Darwin)
  # On Mac, need to also assert name of library with "@rpath" prefix
  INSTNAME := -install_name @rpath/libqiv.$(SHEXT)
else
  INSTNAME :=
endif

testTEEM := $(if $(TEEM_INSTALL),,\
$(warning *)\
$(warning *)\
$(warning * Environment variable TEEM_INSTALL is not set. Set this to the path)\
$(warning * of your teem install, as created by Cmake.)\
$(warning *)\
$(warning *)\
$(error Make quitting))
$(testTEEM)

## NOTE: If you want to change the optimization level to -O0 for debugging,
## you can compile with:
##     make DASHO=
## so that DASHO is set to an empty string prior to processing the Makefile
## contents. On the other hand, in GNU make, ":=" with "override" means
## append to any existing definition. So: if you compile with:
##     make CFLAGS=-DNDEBUG
## then "-DNDEBUG" will be added to the end of CFLAGS, and the C preprocessor
## will then evaluate "#ifdef NDEBUG" to true (or "#ifndef NDEBUG" to false).
## That will make the vcrVerbose variable to be #define'd (at compile-time)
## to 0, which will turn off debugging messages conditioned on vcrVerbose.
##
## To make real==double; compile with:
##     make clean; make CFLAGS=-DQIV_REAL_IS_DOUBLE=1

CC ?= clang
DASHO ?= -O2
DASHG ?= -g
# this way, "make CFLAGS=FOO" will put FOO *after* -Wall -Wextra, etc
# e.g. "make CC=clang CFLAGS=-fsanitize=undefined" to detect undefined behavior
# -DQIV_COMPILE is needed to compile the library in way that allows the same
# header to by used by cffi
override CFLAGS := -Wall -Wextra -std=c99 -DQIV_COMPILE -pedantic $(DASHO) $(DASHG) $(CFLAGS)
# can set can DASHLTPZ to full path to libtpz.a to bypass linker defaults
IPATH ?= $(TEEM_INSTALL)/include
LPATH ?= $(TEEM_INSTALL)/lib
ARRR ?= ar crs

## These are the header and source files involved.
HDR := qiv.h qivPrivate.h qivMath.h
SRCS := misc.c aenum.c field.c ctx.c convo.c
OBJS := $(SRCS:.c=.o)

all: $(LLIBSH)

## all objects depend on headers
$(OBJS): $(HDR)

## from .c to .o files
$(OBJS): %.o: %.c
	$(CC) -c $(CFLAGS) -fPIC -I $(IPATH) -o $@ $<

## collect object files into shared library
## TODO: set rpath
$(LLIBSH): $(OBJS)
	$(CC) -shared $(CFLAGS) -o $(LLIBSH) $(OBJS) $(INSTNAME) \
	  -L $(LPATH) -lteem -lpng -lz -lm

clean:
	rm -f $(OBJS) $(LLIBSH)

# check that library contains what it should and no more:
# nm libqiv.* | grep -v " U " | grep -v " s " | grep -v " d " | grep -v " t "
