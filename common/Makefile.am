SUBDIRS=rng
noinst_LTLIBRARIES = libvprec_tools.la libvfc_hashmap.la libinterflop_fma.la


libvprec_tools_la_CFLAGS = -fPIC -fno-stack-protector
libvprec_tools_la_LDFLAGS = -lquadmath -lm
if WALL_CFLAGS
libvprec_tools_la_CFLAGS += -Wall -Wextra 
endif
libvprec_tools_la_SOURCES = \
    vprec_tools.h \
    vprec_tools.c

libvfc_hashmap_la_CFLAGS = -fPIC -fno-stack-protector
if WALL_CFLAGS
libvfc_hashmap_la_CFLAGS += -Wall -Wextra 
endif
libvfc_hashmap_la_SOURCES = \
    vfc_hashmap.h \
    vfc_hashmap.c \
libvfc_hashmap_la_includedir = $(includedir)/

libinterflop_fma_la_CFLAGS = -fPIC -fno-stack-protector
if WALL_CFLAGS
libinterflop_fma_la_CFLAGS += -Wall -Wextra 
endif
libinterflop_fma_la_SOURCES = \
    fmaqApproc.h \
    fmaqApprox.c 
library_includedir = $(includedir)/
