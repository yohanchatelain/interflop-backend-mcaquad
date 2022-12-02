/*****************************************************************************\
 *                                                                           *\
 *  This file is part of the Verificarlo project,                            *\
 *  under the Apache License v2.0 with LLVM Exceptions.                      *\
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.                 *\
 *  See https://llvm.org/LICENSE.txt for license information.                *\
 *                                                                           *\
 *                                                                           *\
 *  Copyright (c) 2015                                                       *\
 *     Universite de Versailles St-Quentin-en-Yvelines                       *\
 *     CMLA, Ecole Normale Superieure de Cachan                              *\
 *                                                                           *\
 *  Copyright (c) 2018                                                       *\
 *     Universite de Versailles St-Quentin-en-Yvelines                       *\
 *                                                                           *\
 *  Copyright (c) 2019-2022                                                  *\
 *     Verificarlo Contributors                                              *\
 *                                                                           *\
 ****************************************************************************/
// Changelog:
//
// 2015-05-20 replace random number generator with TinyMT64. This
// provides a reentrant, independent generator of better quality than
// the one provided in libc.
//
// 2015-10-11 New version based on quad floating point type to replace MPFR
// until required MCA precision is lower than quad mantissa divided by 2,
// i.e. 56 bits
//
// 2015-11-16 New version using double precision for single precision operation
//
// 2016-07-14 Support denormalized numbers
//
// 2017-04-25 Rewrite debug and validate the noise addition operation
//
// 2019-08-07 Fix memory leak and convert to interflop
//
// 2020-02-07 create separated virtual precisions for binary32
// and binary64. Uses the binary128 structure for easily manipulating bits
// through bitfields. Removes useless specials cases in qnoise and pow2d.
// Change return type from int to void for some functions and uses instead
// errx and warnx for handling errors.
//
// 2020-02-26 Factorize _inexact function into the _INEXACT macro function.
// Use variables for options name instead of hardcoded one.
// Add DAZ/FTZ support.
//
// 2021-10-13 Switched random number generator from TinyMT64 to the one
// provided by the libc. The backend is now re-entrant. Pthread and OpenMP
// threads are now supported.
// Generation of hook functions is now done through macros, shared accross
// backends.
//
// 2022-02-16 Add interflop_user_call implementation for INTERFLOP_CALL_INEXACT
// id Add FAST_INEXACT macro that is a fast version of the INEXACT macro that
// does not check if the number is representable or not and thus always
// introduces a perturbation (for relativeError only)

#include <argp.h>
#include <err.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>

#include "interflop-stdlib/common/float_const.h"
#include "interflop-stdlib/common/float_struct.h"
#include "interflop-stdlib/common/float_utils.h"
#include "interflop-stdlib/common/options.h"
#include "interflop-stdlib/fma/fmaqApprox.h"
#include "interflop-stdlib/interflop.h"
#include "interflop-stdlib/iostream/logger.h"
#include "interflop-stdlib/rng/vfc_rng.h"
#include "interflop_mca.h"

typedef enum {
  KEY_PREC_B32,
  KEY_PREC_B64,
  KEY_ERR_EXP,
  KEY_MODE = 'm',
  KEY_ERR_MODE = 'e',
  KEY_SEED = 's',
  KEY_DAZ = 'd',
  KEY_FTZ = 'f',
  KEY_SPARSITY = 'n'
} key_args;

static const char key_prec_b32_str[] = "precision-binary32";
static const char key_prec_b64_str[] = "precision-binary64";
static const char key_mode_str[] = "mode";
static const char key_err_mode_str[] = "error-mode";
static const char key_err_exp_str[] = "max-abs-error-exponent";
static const char key_seed_str[] = "seed";
static const char key_daz_str[] = "daz";
static const char key_ftz_str[] = "ftz";
static const char key_sparsity_str[] = "sparsity";

static const char *MCA_MODE_STR[] = {[mcamode_ieee] = "ieee",
                                     [mcamode_mca] = "mca",
                                     [mcamode_pb] = "pb",
                                     [mcamode_rr] = "rr"};

/* define the available error modes */
typedef enum {
  mca_err_mode_rel,
  mca_err_mode_abs,
  mca_err_mode_all
} mca_err_mode;

static const char *MCA_ERR_MODE_STR[] = {[mca_err_mode_rel] = "rel",
                                         [mca_err_mode_abs] = "abs",
                                         [mca_err_mode_all] = "all"};

/* define default environment variables and default parameters */
#define MCA_PRECISION_BINARY32_MIN 1
#define MCA_PRECISION_BINARY64_MIN 1
#define MCA_PRECISION_BINARY32_MAX DOUBLE_PMAN_SIZE
#define MCA_PRECISION_BINARY64_MAX QUAD_PMAN_SIZE
#define MCA_PRECISION_BINARY32_DEFAULT FLOAT_PREC
#define MCA_PRECISION_BINARY64_DEFAULT DOUBLE_PREC
#define MCA_MODE_DEFAULT mcamode_mca

/* possible operations values */
typedef enum {
  mca_add = '+',
  mca_sub = '-',
  mca_mul = '*',
  mca_div = '/',
  mca_fma = 'f',
  mca_cast = 'c',
  __mca_operations_end__
} mca_operations;

/******************** MCA CONTROL FUNCTIONS *******************
 * The following functions are used to set virtual precision and
 * MCA mode of operation.
 ***************************************************************/

/* Set the mca mode */
static void _set_mca_mode(const mcamode mode, mcaquad_context_t *ctx) {
  if (mode >= _mcamode_end_) {
    logger_error("--%s invalid value provided, must be one of: "
                 "{ieee, mca, pb, rr}.",
                 key_mode_str);
  }
  ctx->mode = mode;
}

/* Set the virtual precision for binary32 */
static void _set_mca_precision_binary32(const int precision,
                                        mcaquad_context_t *ctx) {
  _set_precision(MCA, precision, ctx->binary32_precision, (float)0);
}

/* Set the virtual precision for binary64 */
static void _set_mca_precision_binary64(const int precision,
                                        mcaquad_context_t *ctx) {
  _set_precision(MCA, precision, ctx->binary64_precision, (double)0);
}

/******************** MCA RANDOM FUNCTIONS ********************
 * The following functions are used to calculate the random
 * perturbations used for MCA
 ***************************************************************/

/* global thread identifier */
pid_t global_tid = 0;

/* helper data structure to centralize the data used for random number
 * generation */
static __thread rng_state_t rng_state;

/* noise = rand * 2^(exp) */
/* We can skip special cases since we never meet them */
/* Since we have exponent of float values, the result */
/* is comprised between: */
/* 127+127 = 254 < DOUBLE_EXP_MAX (1023)  */
/* -126-24+-126-24 = -300 > DOUBLE_EXP_MIN (-1022) */
double _noise_binary64(const int exp, rng_state_t *rng_state);
inline double _noise_binary64(const int exp, rng_state_t *rng_state) {
  const double d_rand = get_rand_double01(rng_state, &global_tid) - 0.5;
  binary64 b64 = {.f64 = d_rand};
  b64.ieee.exponent = b64.ieee.exponent + exp;
  return b64.f64;
}

/* noise = rand * 2^(exp) */
/* We can skip special cases since we never met them */
/* Since we have exponent of double values, the result */
/* is comprised between: */
/* 1023+1023 = 2046 < QUAD_EXP_MAX (16383)  */
/* -1022-53+-1022-53 = -2200 > QUAD_EXP_MIN (-16382) */
__float128 _noise_binary128(const int exp, rng_state_t *rng_state) {
  /* random number in (-0.5, 0.5) */
  const __float128 noise =
      (__float128)get_rand_double01(rng_state, &global_tid) - 0.5Q;
  binary128 b128 = {.f128 = noise};
  b128.ieee128.exponent = b128.ieee128.exponent + exp;
  return b128.f128;
}

#define _IS_IEEE_MODE(CTX)                                                     \
  /* if mode ieee, do not introduce noise */                                   \
  (CTX->mode == mcamode_ieee)

#define _IS_NOT_NORMAL_OR_SUBNORMAL(X)                                         \
  /* Check that we are not in a special case */                                \
  (FPCLASSIFY(X) != FP_NORMAL && FPCLASSIFY(X) != FP_SUBNORMAL)

/* Macro function for checking if the value X must be noised */
#define _MUST_NOT_BE_NOISED(X, VIRTUAL_PRECISION, CTX)                         \
  /* if mode ieee, do not introduce noise */                                   \
  (CTX->mode == mcamode_ieee) ||					                                   \
  /* Check that we are not in a special case */				                         \
  (FPCLASSIFY(X) != FP_NORMAL && FPCLASSIFY(X) != FP_SUBNORMAL) ||	           \
  /* In RR if the number is representable in current virtual precision, */     \
  /* do not add any noise if */						                                     \
  (CTX->mode == mcamode_rr && _IS_REPRESENTABLE(X, VIRTUAL_PRECISION))

/* Generic function for computing the mca noise */
#define _NOISE(X, EXP, RNG_STATE)                                              \
  _Generic(X, double                                                           \
           : _noise_binary64, __float128                                       \
           : _noise_binary128)(EXP, RNG_STATE)

/* Fast version of _INEXACT macro that adds noise in relative error.
  Always adds noise even if X is exact or sparsity is enabled.
 */
#define _FAST_INEXACT(X, VIRTUAL_PRECISION, CTX, RNG_STATE)                    \
  {                                                                            \
    mcaquad_context_t *TMP_CTX = (mcaquad_context_t *)CTX;                     \
    if (_IS_IEEE_MODE(TMP_CTX) || _IS_NOT_NORMAL_OR_SUBNORMAL(*X)) {           \
      return;                                                                  \
    }                                                                          \
    _init_rng_state_struct(&RNG_STATE, TMP_CTX->choose_seed,                   \
                           (unsigned long long)(TMP_CTX->seed), false);        \
    const int32_t e_a = GET_EXP_FLT(*X);                                       \
    const int32_t e_n_rel = e_a - (VIRTUAL_PRECISION - 1);                     \
    const typeof(*X) noise_rel = _NOISE(*X, e_n_rel, &RNG_STATE);              \
    *X = *X + noise_rel;                                                       \
  }

/* Macro function that adds mca noise to X
   according to the virtual_precision VIRTUAL_PRECISION */
#define _INEXACT(X, VIRTUAL_PRECISION, CTX, RNG_STATE)                         \
  {                                                                            \
    mcaquad_context_t *TMP_CTX = (mcaquad_context_t *)CTX;                     \
    _init_rng_state_struct(&RNG_STATE, TMP_CTX->choose_seed,                   \
                           (unsigned long long)(TMP_CTX->seed), false);        \
    if (_MUST_NOT_BE_NOISED(*X, VIRTUAL_PRECISION, TMP_CTX)) {                 \
      return;                                                                  \
    } else if (_mca_skip_eval(TMP_CTX->sparsity, &(RNG_STATE), &global_tid)) { \
      return;                                                                  \
    } else {                                                                   \
      if (TMP_CTX->relErr) {                                                   \
        const int32_t e_a = GET_EXP_FLT(*X);                                   \
        const int32_t e_n_rel = e_a - (VIRTUAL_PRECISION - 1);                 \
        const typeof(*X) noise_rel = _NOISE(*X, e_n_rel, &(RNG_STATE));        \
        *X += noise_rel;                                                       \
      }                                                                        \
      if (TMP_CTX->absErr) {                                                   \
        const int32_t e_n_abs = TMP_CTX->absErr_exp;                           \
        const typeof(*X) noise_abs = _NOISE(*X, e_n_abs, &(RNG_STATE));        \
        *X += noise_abs;                                                       \
      }                                                                        \
    }                                                                          \
  }

/* Adds the mca noise to da */
extern void _mca_inexact_binary64(double *da, void *context);
inline void _mca_inexact_binary64(double *da, void *context) {
  mcaquad_context_t *ctx = (mcaquad_context_t *)context;
  _INEXACT(da, ctx->binary32_precision, ctx, rng_state);
}

/* Adds the mca noise to qa */
extern void _mca_inexact_binary128(__float128 *qa, void *context);
inline void _mca_inexact_binary128(__float128 *qa, void *context) {
  mcaquad_context_t *ctx = (mcaquad_context_t *)context;
  _INEXACT(qa, ctx->binary64_precision, ctx, rng_state);
}

/* Generic functions that adds noise to A */
/* The function is choosen depending on the type of X  */
#define _INEXACT_BINARYN(X, A, CTX)                                            \
  _Generic(X, double                                                           \
           : _mca_inexact_binary64, __float128                                 \
           : _mca_inexact_binary128)(A, CTX)

/******************** MCA ARITHMETIC FUNCTIONS ********************
 * The following set of functions perform the MCA operation. Operands
 * are first converted to quad  format (GCC), inbound and outbound
 * perturbations are applied using the _mca_inexact function, and the
 * result converted to the original format for return
 *******************************************************************/

/* perform_ternary_op: applies the ternary operator (op) to (a), (b) and (c) */
/* and stores the result in (res) */
#define PERFORM_UNARY_OP(op, res, a)                                           \
  switch (op) {                                                                \
  case mca_cast:                                                               \
    res = (float)(a);                                                          \
    break;                                                                     \
  default:                                                                     \
    logger_error("invalid operator %c", op);                                   \
  };

/* perform_bin_op: applies the binary operator (op) to (a) and (b) */
/* and stores the result in (res) */
#define PERFORM_BIN_OP(OP, RES, A, B)                                          \
  switch (OP) {                                                                \
  case mca_add:                                                                \
    RES = (A) + (B);                                                           \
    break;                                                                     \
  case mca_mul:                                                                \
    RES = (A) * (B);                                                           \
    break;                                                                     \
  case mca_sub:                                                                \
    RES = (A) - (B);                                                           \
    break;                                                                     \
  case mca_div:                                                                \
    RES = (A) / (B);                                                           \
    break;                                                                     \
  default:                                                                     \
    logger_error("invalid operator %c", OP);                                   \
  };

/* perform_ternary_op: applies the ternary operator (op) to (a), (b) and (c) */
/* and stores the result in (res) */
#define PERFORM_TERNARY_OP(op, res, a, b, c)                                   \
  switch (op) {                                                                \
  case mca_fma:                                                                \
    res = fmaApprox((a), (b), (c));                                            \
    break;                                                                     \
  default:                                                                     \
    logger_error("invalid operator %c", op);                                   \
  };

/* Generic macro function that returns mca(A OP B) */
/* Functions are determined according to the type of X */
#define _MCA_UNARY_OP(A, OP, CTX, X)                                           \
  do {                                                                         \
    typeof(X) _A = A;                                                          \
    typeof(X) _RES = 0;                                                        \
    mcaquad_context_t *TMP_CTX = (mcaquad_context_t *)CTX;                     \
    if (TMP_CTX->daz) {                                                        \
      _A = DAZ(A);                                                             \
    }                                                                          \
    if (TMP_CTX->mode == mcamode_pb || TMP_CTX->mode == mcamode_mca) {         \
      _INEXACT_BINARYN(X, &_A, CTX);                                           \
    }                                                                          \
    PERFORM_UNARY_OP(OP, _RES, _A);                                            \
    if (TMP_CTX->mode == mcamode_rr || TMP_CTX->mode == mcamode_mca) {         \
      _INEXACT_BINARYN(X, &_RES, CTX);                                         \
    }                                                                          \
    if (TMP_CTX->ftz) {                                                        \
      _RES = FTZ((typeof(A))_RES);                                             \
    }                                                                          \
    return (typeof(A))(_RES);                                                  \
  } while (0);

/* Generic macro function that returns mca(A OP B) */
/* Functions are determined according to the type of X */
#define _MCA_BINARY_OP(A, B, OP, CTX, X)                                       \
  do {                                                                         \
    typeof(X) _A = A;                                                          \
    typeof(X) _B = B;                                                          \
    typeof(X) _RES = 0;                                                        \
    mcaquad_context_t *TMP_CTX = (mcaquad_context_t *)CTX;                     \
    if (TMP_CTX->daz) {                                                        \
      _A = DAZ(A);                                                             \
      _B = DAZ(B);                                                             \
    }                                                                          \
    if (TMP_CTX->mode == mcamode_pb || TMP_CTX->mode == mcamode_mca) {         \
      _INEXACT_BINARYN(X, &_A, CTX);                                           \
      _INEXACT_BINARYN(X, &_B, CTX);                                           \
    }                                                                          \
    PERFORM_BIN_OP(OP, _RES, _A, _B);                                          \
    if (TMP_CTX->mode == mcamode_rr || TMP_CTX->mode == mcamode_mca) {         \
      _INEXACT_BINARYN(X, &_RES, CTX);                                         \
    }                                                                          \
    if (TMP_CTX->ftz) {                                                        \
      _RES = FTZ((typeof(A))_RES);                                             \
    }                                                                          \
    return (typeof(A))(_RES);                                                  \
  } while (0);

/* Generic macro function that returns mca(A OP B OP C) */
/* Functions are determined according to the type of X */
#define _MCA_TERNARY_OP(A, B, C, OP, CTX, X)                                   \
  do {                                                                         \
    typeof(X) _A = A;                                                          \
    typeof(X) _B = B;                                                          \
    typeof(X) _C = C;                                                          \
    typeof(X) _RES = 0;                                                        \
    mcaquad_context_t *TMP_CTX = (mcaquad_context_t *)CTX;                     \
    if (TMP_CTX->daz) {                                                        \
      _A = DAZ(A);                                                             \
      _B = DAZ(B);                                                             \
      _C = DAZ(C);                                                             \
    }                                                                          \
    if (TMP_CTX->mode == mcamode_pb || TMP_CTX->mode == mcamode_mca) {         \
      _INEXACT_BINARYN(X, &_A, CTX);                                           \
      _INEXACT_BINARYN(X, &_B, CTX);                                           \
      _INEXACT_BINARYN(X, &_C, CTX);                                           \
    }                                                                          \
    PERFORM_TERNARY_OP(OP, _RES, _A, _B, _C);                                  \
    if (TMP_CTX->mode == mcamode_rr || TMP_CTX->mode == mcamode_mca) {         \
      _INEXACT_BINARYN(X, &_RES, CTX);                                         \
    }                                                                          \
    if (TMP_CTX->ftz) {                                                        \
      _RES = FTZ((typeof(A))_RES);                                             \
    }                                                                          \
    return (typeof(A))(_RES);                                                  \
  } while (0);

/* Performs mca(dop a) where a is a binary32 value */
/* Intermediate computations are performed with binary64 */
inline float _mca_binary32_unary_op(const float a, const mca_operations dop,
                                    void *context) {
  _MCA_UNARY_OP(a, dop, context, (double)0);
}

/* Performs mca(a dop b) where a and b are binary32 values */
/* Intermediate computations are performed with binary64 */
float _mca_binary32_binary_op(const float a, const float b,
                              const mca_operations dop, void *context);
inline float _mca_binary32_binary_op(const float a, const float b,
                                     const mca_operations dop, void *context) {
  _MCA_BINARY_OP(a, b, dop, context, (double)0);
}

/* Performs mca(a dop b dop c) where a, b and c are binary32 values */
/* Intermediate computations are performed with binary64 */
float _mca_binary32_ternary_op(const float a, const float b, const float c,
                               const mca_operations dop, void *context);
inline float _mca_binary32_ternary_op(const float a, const float b,
                                      const float c, const mca_operations dop,
                                      void *context) {
  _MCA_TERNARY_OP(a, b, c, dop, context, (double)0);
}

/* Performs mca(qop a) where a is a binary64 value */
/* Intermediate computations are performed with binary128 */
double _mca_binary64_unary_op(const double a, const mca_operations qop,
                              void *context);
inline double _mca_binary64_unary_op(const double a, const mca_operations qop,
                                     void *context) {
  _MCA_UNARY_OP(a, qop, context, (__float128)0);
}

/* Performs mca(a qop b) where a and b are binary64 values */
/* Intermediate computations are performed with binary128 */
double _mca_binary64_binary_op(const double a, const double b,
                               const mca_operations qop, void *context);
inline double _mca_binary64_binary_op(const double a, const double b,
                                      const mca_operations qop, void *context) {
  _MCA_BINARY_OP(a, b, qop, context, (__float128)0);
}

/* Performs mca(a qop b qop c) where a, b and c are binary64 values */
/* Intermediate computations are performed with binary128 */
double _mca_binary64_ternary_op(const double a, const double b, const double c,
                                const mca_operations qop, void *context);

inline double _mca_binary64_ternary_op(const double a, const double b,
                                       const double c, const mca_operations qop,
                                       void *context) {
  _MCA_TERNARY_OP(a, b, c, qop, context, (__float128)0);
}

/************************* FPHOOKS FUNCTIONS *************************
 * These functions correspond to those inserted into the source code
 * during source to source compilation and are replacement to floating
 * point operators
 **********************************************************************/

void INTERFLOP_MCAQUAD_API(add_float)(float a, float b, float *res,
                                      void *context) {
  *res = _mca_binary32_binary_op(a, b, mca_add, context);
}

void INTERFLOP_MCAQUAD_API(sub_float)(float a, float b, float *res,
                                      void *context) {
  *res = _mca_binary32_binary_op(a, b, mca_sub, context);
}

void INTERFLOP_MCAQUAD_API(mul_float)(float a, float b, float *res,
                                      void *context) {
  *res = _mca_binary32_binary_op(a, b, mca_mul, context);
}

void INTERFLOP_MCAQUAD_API(div_float)(float a, float b, float *res,
                                      void *context) {
  *res = _mca_binary32_binary_op(a, b, mca_div, context);
}

void INTERFLOP_MCAQUAD_API(fma_float)(float a, float b, float c, float *res,
                                      void *context) {
  *res = _mca_binary32_ternary_op(a, b, c, mca_fma, context);
}

void INTERFLOP_MCAQUAD_API(add_double)(double a, double b, double *res,
                                       void *context) {
  *res = _mca_binary64_binary_op(a, b, mca_add, context);
}

void INTERFLOP_MCAQUAD_API(sub_double)(double a, double b, double *res,
                                       void *context) {
  *res = _mca_binary64_binary_op(a, b, mca_sub, context);
}

void INTERFLOP_MCAQUAD_API(mul_double)(double a, double b, double *res,
                                       void *context) {
  *res = _mca_binary64_binary_op(a, b, mca_mul, context);
}

void INTERFLOP_MCAQUAD_API(div_double)(double a, double b, double *res,
                                       void *context) {
  *res = _mca_binary64_binary_op(a, b, mca_div, context);
}

void INTERFLOP_MCAQUAD_API(fma_double)(double a, double b, double c,
                                       double *res, void *context) {
  *res = _mca_binary64_ternary_op(a, b, c, mca_fma, context);
}

void INTERFLOP_MCAQUAD_API(cast_double_to_float)(double a, float *res,
                                                 void *context) {
  *res = _mca_binary64_unary_op(a, mca_cast, context);
}

void _interflop_usercall_inexact(void *context, va_list ap) {
  mcaquad_context_t *ctx = (mcaquad_context_t *)context;
  double xd = 0;
  __float128 xq = 0;
  enum FTYPES ftype;
  void *value = NULL;
  int precision = 0, t = 0;
  ftype = va_arg(ap, enum FTYPES);
  value = va_arg(ap, void *);
  precision = va_arg(ap, int);
  switch (ftype) {
  case FFLOAT:
    xd = *((float *)value);
    t = (precision <= 0) ? (ctx->binary32_precision + precision) : precision;
    _FAST_INEXACT(&xd, t, context, rng_state);
    *((float *)value) = xd;
    break;
  case FDOUBLE:
    xq = *((double *)value);
    t = (precision <= 0) ? (ctx->binary64_precision + precision) : precision;
    _FAST_INEXACT(&xq, t, context, rng_state);
    *((double *)value) = xq;
    break;
  case FQUAD:
    xq = *((__float128 *)value);
    _FAST_INEXACT(&xq, precision, context, rng_state);
    *((__float128 *)value) = xq;
    break;
  default:
    logger_warning(
        "Uknown type passed to _interflop_usercall_inexact function");
    break;
  }
}

void INTERFLOP_MCAQUAD_API(user_call)(void *context, interflop_call_id id,
                                      va_list ap) {
  switch (id) {
  case INTERFLOP_INEXACT_ID:
    _interflop_usercall_inexact(context, ap);
    break;
  case INTERFLOP_SET_PRECISION_BINARY32:
    _set_mca_precision_binary32(va_arg(ap, int), context);
    break;
  case INTERFLOP_SET_PRECISION_BINARY64:
    _set_mca_precision_binary64(va_arg(ap, int), context);
    break;
  default:
    logger_warning("Unknown interflop_call id (=%d)", id);
    break;
  }
}

static struct argp_option options[] = {
    {key_prec_b32_str, KEY_PREC_B32, "PRECISION", 0,
     "select precision for binary32 (PRECISION > 0)", 0},
    {key_prec_b64_str, KEY_PREC_B64, "PRECISION", 0,
     "select precision for binary64 (PRECISION > 0)", 0},
    {key_mode_str, KEY_MODE, "MODE", 0,
     "select MCA mode among {ieee, mca, pb, rr}", 0},
    {key_err_mode_str, KEY_ERR_MODE, "ERROR_MODE", 0,
     "select error mode among {rel, abs, all}", 0},
    {key_err_exp_str, KEY_ERR_EXP, "MAX_ABS_ERROR_EXPONENT", 0,
     "select magnitude of the maximum absolute error", 0},
    {key_seed_str, KEY_SEED, "SEED", 0, "fix the random generator seed", 0},
    {key_daz_str, KEY_DAZ, 0, 0,
     "denormals-are-zero: sets denormals inputs to zero", 0},
    {key_ftz_str, KEY_FTZ, 0, 0, "flush-to-zero: sets denormal output to zero",
     0},
    {key_sparsity_str, KEY_SPARSITY, "SPARSITY", 0,
     "one in {sparsity} operations will be perturbed. 0 < sparsity <= 1.", 0},
    {0}};

error_t parse_opt(int key, char *arg, struct argp_state *state) {
  mcaquad_context_t *ctx = (mcaquad_context_t *)state->input;
  char *endptr;
  int val = -1;
  int error = 0;
  switch (key) {
  case KEY_PREC_B32:
    /* precision for binary32 */
    error = 0;
    val = interflop_strtol(arg, &endptr, &error);
    if (error != 0 || val <= 0) {
      logger_error("--%s invalid value provided, must be a positive integer",
                   key_prec_b32_str);
    } else {
      _set_mca_precision_binary32(val, ctx);
    }
    break;
  case KEY_PREC_B64:
    /* precision for binary64 */
    error = 0;
    val = interflop_strtol(arg, &endptr, &error);
    if (error != 0 || val <= 0) {
      logger_error("--%s invalid value provided, must be a positive integer",
                   key_prec_b64_str);
    } else {
      _set_mca_precision_binary64(val, ctx);
    }
    break;
  case KEY_MODE:
    /* mca mode */
    if (interflop_strcasecmp(MCA_MODE_STR[mcamode_ieee], arg) == 0) {
      _set_mca_mode(mcamode_ieee, ctx);
    } else if (interflop_strcasecmp(MCA_MODE_STR[mcamode_mca], arg) == 0) {
      _set_mca_mode(mcamode_mca, ctx);
    } else if (interflop_strcasecmp(MCA_MODE_STR[mcamode_pb], arg) == 0) {
      _set_mca_mode(mcamode_pb, ctx);
    } else if (interflop_strcasecmp(MCA_MODE_STR[mcamode_rr], arg) == 0) {
      _set_mca_mode(mcamode_rr, ctx);
    } else {
      logger_error("--%s invalid value provided, must be one of: "
                   "{ieee, mca, pb, rr}.",
                   key_mode_str);
    }
    break;
  case KEY_ERR_MODE:
    /* mca error mode */
    if (interflop_strcasecmp(MCA_ERR_MODE_STR[mca_err_mode_rel], arg) == 0) {
      ctx->relErr = true;
      ctx->absErr = false;
    } else if (interflop_strcasecmp(MCA_ERR_MODE_STR[mca_err_mode_abs], arg) ==
               0) {
      ctx->relErr = false;
      ctx->absErr = true;
    } else if (interflop_strcasecmp(MCA_ERR_MODE_STR[mca_err_mode_all], arg) ==
               0) {
      ctx->relErr = true;
      ctx->absErr = true;
    } else {
      logger_error("--%s invalid value provided, must be one of: "
                   "{rel, abs, all}.",
                   key_err_mode_str);
    }
    break;
  case KEY_ERR_EXP:
    /* exponent of the maximum absolute error */
    error = 0;
    ctx->absErr_exp = interflop_strtol(arg, &endptr, &error);
    if (error != 0) {
      logger_error("--%s invalid value provided, must be an integer",
                   key_err_exp_str);
    }
    break;
  case KEY_SEED:
    /* seed */
    error = 0;
    ctx->choose_seed = true;
    ctx->seed = interflop_strtol(arg, &endptr, &error);
    if (error != 0) {
      logger_error("--%s invalid value provided, must be an integer",
                   key_seed_str);
    }
    break;
  case KEY_DAZ:
    /* denormals-are-zero */
    ctx->daz = true;
    break;
  case KEY_FTZ:
    /* flush-to-zero */
    ctx->ftz = true;
    break;
  case KEY_SPARSITY:
    /* sparse perturbations */
    error = 0;
    ctx->sparsity = interflop_strtod(arg, &endptr, &error);
    if (ctx->sparsity <= 0) {
      error = 1;
    }
    if (error != 0) {
      logger_error("--%s invalid value provided, must be positive",
                   key_sparsity_str);
    }
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

struct argp argp = {options, parse_opt, "", "", NULL, NULL, NULL};

void init_context(mcaquad_context_t *ctx) {
  ctx->mode = MCA_MODE_DEFAULT;
  ctx->binary32_precision = MCA_PRECISION_BINARY32_DEFAULT;
  ctx->binary64_precision = MCA_PRECISION_BINARY64_DEFAULT;
  ctx->relErr = true;
  ctx->absErr = false;
  ctx->absErr_exp = 112;
  ctx->choose_seed = false;
  ctx->daz = false;
  ctx->ftz = false;
  ctx->seed = 0ULL;
  ctx->sparsity = 1.0f;
}

void print_information_header(void *context) {
  mcaquad_context_t *ctx = (mcaquad_context_t *)context;

  logger_info(
      "load backend with "
      "%s = %d, "
      "%s = %d, "
      "%s = %s, "
      "%s = %s, "
      "%s = %d, "
      "%s = %s, "
      "%s = %s and "
      "%s = %f"
      "\n",
      key_prec_b32_str, ctx->binary32_precision, key_prec_b64_str,
      ctx->binary64_precision, key_mode_str, MCA_MODE_STR[ctx->mode],
      key_err_mode_str,
      (ctx->relErr && !ctx->absErr)   ? MCA_ERR_MODE_STR[mca_err_mode_rel]
      : (!ctx->relErr && ctx->absErr) ? MCA_ERR_MODE_STR[mca_err_mode_abs]
      : (ctx->relErr && ctx->absErr)  ? MCA_ERR_MODE_STR[mca_err_mode_all]
                                      : MCA_ERR_MODE_STR[mca_err_mode_rel],
      key_err_exp_str, (ctx->absErr_exp), key_daz_str,
      ctx->daz ? "true" : "false", key_ftz_str, ctx->ftz ? "true" : "false",
      key_sparsity_str, ctx->sparsity);
}

void INTERFLOP_MCAQUAD_API(CLI)(int argc, char **argv, void *context) {
  /* parse backend arguments */
  mcaquad_context_t *ctx = (mcaquad_context_t *)context;
  if (interflop_argp_parse != NULL) {
    interflop_argp_parse(&argp, argc, argv, 0, 0, ctx);
  } else {
    interflop_panic("Interflop backend error: argp_parse not implemented\n"
                    "Provide implementation or use interflop_configure to "
                    "configure the backend\n");
  }
}

#define CHECK_IMPL(name)                                                       \
  if (interflop_##name == Null) {                                              \
    interflop_panic("Interflop backend error: " #name " not implemented\n");   \
  }

void _mcaquad_check_stdlib(void) {
  CHECK_IMPL(malloc);
  CHECK_IMPL(exit);
  CHECK_IMPL(fopen);
  CHECK_IMPL(fprintf);
  CHECK_IMPL(getenv);
  CHECK_IMPL(gettid);
  CHECK_IMPL(sprintf);
  CHECK_IMPL(strcasecmp);
  CHECK_IMPL(strerror);
  CHECK_IMPL(vfprintf);
  CHECK_IMPL(vwarnx);
}

void INTERFLOP_MCAQUAD_API(pre_init)(File *stream, interflop_panic_t panic,
                                     void **context) {
  interflop_set_handler("panic", panic);
  _mcaquad_check_stdlib();

  /* Initialize the logger */
  logger_init(stream);

  /* allocate the context */
  mcaquad_context_t *ctx =
      (mcaquad_context_t *)interflop_malloc(sizeof(mcaquad_context_t));
  init_context(ctx);
  *context = ctx;
}

struct interflop_backend_interface_t
INTERFLOP_MCAQUAD_API(init)(void *context) {

  mcaquad_context_t *ctx = (mcaquad_context_t *)context;

  print_information_header(ctx);

  struct interflop_backend_interface_t interflop_backend_mcaquad = {
    interflop_add_float : INTERFLOP_MCAQUAD_API(add_float),
    interflop_sub_float : INTERFLOP_MCAQUAD_API(sub_float),
    interflop_mul_float : INTERFLOP_MCAQUAD_API(mul_float),
    interflop_div_float : INTERFLOP_MCAQUAD_API(div_float),
    interflop_cmp_float : NULL,
    interflop_add_double : INTERFLOP_MCAQUAD_API(add_double),
    interflop_sub_double : INTERFLOP_MCAQUAD_API(sub_double),
    interflop_mul_double : INTERFLOP_MCAQUAD_API(mul_double),
    interflop_div_double : INTERFLOP_MCAQUAD_API(div_double),
    interflop_cmp_double : NULL,
    interflop_cast_double_to_float :
        INTERFLOP_MCAQUAD_API(cast_double_to_float),
    interflop_fma_float : INTERFLOP_MCAQUAD_API(fma_float),
    interflop_fma_double : INTERFLOP_MCAQUAD_API(fma_double),
    interflop_enter_function : NULL,
    interflop_exit_function : NULL,
    interflop_user_call : INTERFLOP_MCAQUAD_API(user_call),
    interflop_finalize : NULL,
  };

  /* The seed for the RNG is initialized upon the first request for a random
     number */

  _init_rng_state_struct(&rng_state, ctx->choose_seed, ctx->seed, false);

  return interflop_backend_mcaquad;
}

struct interflop_backend_interface_t interflop_init(void *context)
    __attribute__((weak, alias("interflop_mcaquad_init")));

void interflop_pre_init(File *stream, interflop_panic_t panic, void **context)
    __attribute__((weak, alias("interflop_mcaquad_pre_init")));

void interflop_CLI(int argc, char **argv, void *context)
    __attribute__((weak, alias("interflop_mcaquad_CLI")));
