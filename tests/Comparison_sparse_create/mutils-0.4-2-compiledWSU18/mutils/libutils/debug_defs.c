/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "config.h"
#include "debug_defs.h"

static int debug = 0;

int get_debug_mode(void)
{
  return debug;
}

void set_debug_mode(int d)
{
  debug = d;
}
