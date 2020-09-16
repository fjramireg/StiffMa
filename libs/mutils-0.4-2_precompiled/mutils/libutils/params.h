/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef PARAMS_H
#define PARAMS_H

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "memutils.h"

  void params_free(void);
  void param_print_help(int argc, char **argv);
  int param_get(int argc, char **argv, const char *param, const char *help_str);
  int param_get_int(int argc, char **argv, const char *param, const char *help_str, int idefault);
  double param_get_double(int argc, char **argv, const char *param, const char *help_str, double ddefault);
  const char *param_get_string(int argc, char **argv, const char *param, const char *help_str);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
