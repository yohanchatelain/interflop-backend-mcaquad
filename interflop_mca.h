/*****************************************************************************\
 *                                                                           *\
 *  This file is part of the Verificarlo project,                            *\
 *  under the Apache License v2.0 with LLVM Exceptions.                      *\
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.                 *\
 *  See https://llvm.org/LICENSE.txt for license information.                *\
 *                                                                           *\
 *  Copyright (c) 2019-2022                                                  *\
 *     Verificarlo Contributors                                              *\
 *                                                                           *\
 ****************************************************************************/

#ifndef __INTERFLOP_MCAQUAD_H__
#define __INTERFLOP_MCAQUAD_H__

#include "interflop-stdlib/interflop_stdlib.h"

#define INTERFLOP_MCAQUAD_API(name) interflop_mcaquad_##name

/* define default environment variables and default parameters */
#define MCAQUAD_PRECISION_BINARY32_MIN 1
#define MCAQUAD_PRECISION_BINARY64_MIN 1
#define MCAQUAD_PRECISION_BINARY32_MAX DOUBLE_PMAN_SIZE
#define MCAQUAD_PRECISION_BINARY64_MAX QUAD_PMAN_SIZE
#define MCAQUAD_PRECISION_BINARY32_DEFAULT FLOAT_PREC
#define MCAQUAD_PRECISION_BINARY64_DEFAULT DOUBLE_PREC
#define MCAQUAD_MODE_DEFAULT mcaquad_mode_mca
#define MCAQUAD_ERR_MODE_DEFAULT mcaquad_err_mode_rel

/* define the available MCA modes of operation */
typedef enum {
  mcaquad_mode_ieee,
  mcaquad_mode_mca,
  mcaquad_mode_pb,
  mcaquad_mode_rr,
  _mcaquad_mode_end_
} mcaquad_mode;

/* define the available error modes */
typedef enum {
  mcaquad_err_mode_rel,
  mcaquad_err_mode_abs,
  mcaquad_err_mode_all
} mcaquad_err_mode;

/* Interflop context */
typedef struct {
  IUint64_t seed;
  float sparsity;
  int binary32_precision;
  int binary64_precision;
  int absErr_exp;
  IBool relErr;
  IBool absErr;
  IBool daz;
  IBool ftz;
  IBool choose_seed;
  mcaquad_mode mode;
} mcaquad_context_t;

typedef struct {
  IUint64_t seed;
  float sparsity;
  IUint32_t precision_binary32;
  IUint32_t precision_binary64;
  mcaquad_mode mode;
  mcaquad_err_mode err_mode;
  IInt64_t max_abs_err_exponent;
  IUint32_t daz;
  IUint32_t ftz;
} mcaquad_conf_t;

#endif /* __INTERFLOP_MCAQUAD_H__ */