#
# Makefile for LibNC
# Copyright (c) 2018-2021 Fabrice Bellard
#
#CONFIG_WIN32=y
#CONFIG_ASAN=y

ifdef CONFIG_WIN32
CROSS_PREFIX=x86_64-w64-mingw32-
EXE=.exe
LIBEXT=.lib
DLLEXT=.dll
else
LIBEXT=.a
DLLEXT=.so
endif

HOST_CC=gcc
CC=$(CROSS_PREFIX)gcc
AR=$(CROSS_PREFIX)ar
CFLAGS_VERSION:=-DCONFIG_VERSION=\"$(shell cat VERSION)\"
CFLAGS=-O3 -Wall -Wpointer-arith -g -fno-math-errno -fno-trapping-math -MMD -Wno-format-truncation $(CFLAGS_VERSION)
LDFLAGS=-Wl,-rpath='$$ORIGIN/'
PROGS=nctest$(EXE) ncspeed$(EXE) matmul_test$(EXE) dump_coefs$(EXE)
LIBS+=-lm -lpthread

ifdef CONFIG_ASAN
CFLAGS+=-fsanitize=address -fno-omit-frame-pointer
LDFLAGS+=-fsanitize=address -fno-omit-frame-pointer
endif

all: $(PROGS)

clean:
	rm -f *.o *.d $(PROGS)

nctest$(EXE): nctest.o libnc$(DLLEXT)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

ncspeed$(EXE): ncspeed.o libnc$(DLLEXT)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

# needed to enable FMA
matmul_test.o: CFLAGS+=-march=haswell

matmul_test$(EXE): matmul_test.o libnc$(DLLEXT)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

dump_coefs$(EXE): dump_coefs.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

-include $(wildcard *.d)
