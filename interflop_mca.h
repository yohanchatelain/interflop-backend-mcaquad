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

#include "interflop_stdlib.h"

#define INTERFLOP_MCAQUAD_API(name) interflop_mcaquad_##name

/* define the available MCA modes of operation */
typedef enum {
  mcamode_ieee,
  mcamode_mca,
  mcamode_pb,
  mcamode_rr,
  _mcamode_end_
} mcamode;

/* Interflop context */
typedef struct {
  IBool relErr;
  IBool absErr;
  IBool daz;
  IBool ftz;
  IBool choose_seed;
  mcamode mode;
  int binary32_precision;
  int binary64_precision;
  int absErr_exp;
  float sparsity;
  IUint64_t seed;
} mcaquad_context_t;

#endif /* __INTERFLOP_MCAQUAD_H__ */