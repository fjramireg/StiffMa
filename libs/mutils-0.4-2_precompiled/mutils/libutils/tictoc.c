/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "tictoc.h"
#include "debug_defs.h"

static double flops = 0;
static double bytes = 0;
static long time_us = 0;

#ifdef WINDOWS
#else
static struct timeval tb, te;

#ifndef APPLE
#ifdef USE_OPENMP
#pragma omp threadprivate(tb, te, flops, bytes)
#endif /* USE_OPENMP */
#endif /* APPLE */

#endif



#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif

void stats_zero(void)
{
  flops = 0;
  time_us = 0;
}

void flops_add(double nflops)
{
  flops += nflops;
}

void bytes_add(double nbytes)
{
  bytes += nbytes;
}

double flops_get()
{
  return flops;
}

double bytes_get()
{
  return bytes;
}


#ifdef WINDOWS
double elapsed_time(){}
void _tic(){}
void _toc(){}
void _midtoc(){}
void _ntoc(const char *idtxt){}
void _nntoc(){}
void _inctime(){}
void stats_print(){}
#else

double elapsed_time()
{
  long s,u;
  double tt;
  gettimeofday(&te, NULL);
  s=te.tv_sec-tb.tv_sec;
  u=te.tv_usec-tb.tv_usec;
  tt=((double)s)*1000000+u;
  return tt/1e6;
}

void _tic()
{
#ifdef USE_OPENMP
#pragma omp master
#endif
  {
    gettimeofday(&tb, NULL);
    flops=0;
    bytes=0;
    fflush(stdout);
  }
}

void _toc()
{
#ifdef USE_OPENMP
#pragma omp master
#endif
  {
    long s,u;
    double tt;
    gettimeofday(&te, NULL);
    s=te.tv_sec-tb.tv_sec;
    u=te.tv_usec-tb.tv_usec;
    tt=((double)s)*1000000+u;
    VERBOSE("time:                  %li.%.6lis", DEBUG_BASIC, (s*1000000+u)/1000000, (s*1000000+u)%1000000);
    VERBOSE("MFLOP/s:               %.3lf", DEBUG_BASIC, flops/tt);
    VERBOSE("MB/s:                  %.3lf", DEBUG_BASIC, bytes/tt);
    VERBOSE("total fp operations:   %.0lf", DEBUG_BASIC, flops);
    VERBOSE("total memory traffic   %.0lf", DEBUG_BASIC, bytes);
    fflush(stdout);
  }
}

void _midtoc()
{
#ifdef USE_OPENMP
#pragma omp master
#endif
  {
    long s,u;
    double tt;
    gettimeofday(&te, NULL);
    s=te.tv_sec-tb.tv_sec;
    u=te.tv_usec-tb.tv_usec;
    tt=((double)s)*1000000+u;
    VERBOSE("time:                  %li.%.6lis", DEBUG_BASIC, (s*1000000+u)/1000000, (s*1000000+u)%1000000);
    VERBOSE("MFLOP/s:               %.3lf\n", DEBUG_BASIC, flops/tt);
    fflush(stdout);
  }
}

void _ntoc(const char *idtxt)
{
#ifdef USE_OPENMP
#pragma omp master
#endif
  {
    long s,u;
    gettimeofday(&te, NULL);
    s=te.tv_sec-tb.tv_sec;
    u=te.tv_usec-tb.tv_usec;
    if(idtxt){
      VERBOSE("%-30s%10li.%.6lis", DEBUG_BASIC, idtxt, (s*1000000+u)/1000000, (s*1000000+u)%1000000);
    } else {
      VERBOSE("time:%10li.%.6lis", DEBUG_BASIC, (s*1000000+u)/1000000, (s*1000000+u)%1000000);
    }
    fflush(stdout);
  }
}

void _inctime()
{
  gettimeofday(&te, NULL);
  time_us += (te.tv_sec-tb.tv_sec)*1000000 + (te.tv_usec-tb.tv_usec);
}

void stats_print()
{
  VERBOSE("total time %li.%6lis", DEBUG_BASIC, time_us/1000000, time_us%1000000);
}

#endif /* WINDOWS */


