lib_LTLIBRARIES = libinterflop_mca.la
libinterflop_mca_la_SOURCES = interflop_mca.c common/logger.c common/options.c common/fmaqApprox.c
libinterflop_mca_la_CFLAGS = -DBACKEND_HEADER="interflop_mca" -I@INTERFLOP_STDLIB_PATH@/include/
if WALL_CFLAGS
libinterflop_mca_la_CFLAGS += -Wall -Wextra
endif
libinterflop_mca_la_LDFLAGS = -lm
libinterflop_mca_la_LIBADD = @INTERFLOP_STDLIB_PATH@/lib/libinterflop_rng.la
if BUILD_INTERFLOP_STDLIB
libinterflop_mca_la_LIBADD += @INTERFLOP_STDLIB_PATH@/lib/libinterflop_stdlib.la
endif
library_includedir =$(includedir)/
