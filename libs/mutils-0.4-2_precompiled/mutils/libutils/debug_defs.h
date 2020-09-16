/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef DEBUG_DEFS_H
#define DEBUG_DEFS_H

#include "config.h"
#include <stdlib.h>
#include <stdio.h>

#include "message_id.h"

#ifndef WINDOWS
#include <libgen.h>
#endif

#define DEBUG_NONE 0
#define DEBUG_BASIC 1
#define DEBUG_DETAILED 5
#define DEBUG_MEMORY 6
#define DEBUG_ALL 10

#ifdef _MSC_VER
#define DO_PRAGMA(string) __pragma(string)
#else
#define DO_PRAGMA(string) _Pragma(string)
#endif

#ifdef DEBUG_COMPILE
#ifdef _MSC_VER
#define COMPILE_MESSAGE(msg) __pragma(message (msg))
#elif GCC_VERSION >= 40400
#define _PRAGMA(msg) _Pragma(#msg)
#define COMPILE_MESSAGE(msg) _PRAGMA(message #msg)
#else
#warning "Compile-time debugging not supported with this compiler."
#define COMPILE_MESSAGE(msg)
#endif
#else
#define COMPILE_MESSAGE(msg)
#endif


#ifdef MATLAB_MEX_FILE
#include <mex.h>
#define WARNING_HDR ""
#define ERROR_HDR ""
#define MSG_NEWLINE ""

#ifdef USE_OPENMP

/* 
   MEX is not thread-safe. These macros introduce
   an explicit OMP CRITICAL region for the mex calls.
   Works for C99.
 */

#define PRINTF_MSG(msg)				\
  {						\
    DO_PRAGMA("omp critical(print_msg)");	\
    mexPrintf("%s", msg);			\
  }

#define PRINTF_ERR(msg, id)			\
  {						\
    DO_PRAGMA("omp critical(print_err)");	\
    mexErrMsgIdAndTxt(id, msg);			\
  }

#define PRINTF_WRN(msg, id)			\
  {						\
    DO_PRAGMA("omp critical(print_wrn)");	\
    mexWarnMsgIdAndTxt(id, msg);		\
  }
#else /* USE_OPENMP */

/* assuming no multi-threading */
#define PRINTF_MSG(msg)     mexPrintf("%s", msg)
#define PRINTF_ERR(msg, id) mexErrMsgIdAndTxt(id, msg)
#define PRINTF_WRN(msg, id) mexWarnMsgIdAndTxt(id, msg)

#endif /* USE_OPENMP */

#else /* MATLAB_MEX_FILE */
#define WARNING_HDR " ** WARNING: "
#define ERROR_HDR " ** ERROR: "
#define MSG_NEWLINE "\n"
#define PRINTF_MSG(msg) printf("%s", msg)
#define PRINTF_ERR(msg, id) {fprintf(stderr, "%s", msg); exit(0);}
#define PRINTF_WRN(msg, id) printf("%s %s", id, msg)
#endif /* MATLAB_MEX_FILE */


#undef VERBOSE
#define VERBOSE(msg, level, ...) {				\
    if(level<=get_debug_mode()){				\
      char __buff[256];						\
      SNPRINTF(__buff, 255, " ** " msg "\n", ##__VA_ARGS__);	\
      __buff[255] = 0;						\
      PRINTF_MSG(__buff);fflush(stdout);			\
    }								\
  }

#undef MESSAGE
#define MESSAGE(msg, ...) VERBOSE(msg, 0, ##__VA_ARGS__)

#undef EMESSAGE
#define EMESSAGE(msg, ...) {						\
    char __buff[256];							\
    SNPRINTF(__buff, 255, ERROR_HDR "%s: %s(): %d: " msg MSG_NEWLINE, BASENAME(__FILE__), __FUNCTION__, __LINE__, ##__VA_ARGS__); \
    __buff[255] = 0;							\
    PRINTF_ERR(__buff, MUTILS_INTERNAL_ERROR);				\
  }

#undef WMESSAGE
#define WMESSAGE(msg, ...) {						\
    char __buff[256];							\
    SNPRINTF(__buff, 255, WARNING_HDR " %s: %s(): %d: " msg MSG_NEWLINE, BASENAME(__FILE__), __FUNCTION__, __LINE__, ##__VA_ARGS__); \
    __buff[255] = 0;							\
    PRINTF_MSG(__buff);							\
  }

#undef ERROR
#define ERROR(msg,...) {			\
    EMESSAGE(msg, ##__VA_ARGS__);		\
    exit(1);					\
  }

#undef WARNING
#define WARNING(msg, ...) {			\
    WMESSAGE(msg, ##__VA_ARGS__);		\
  }

/* User-versions for mex files */
#undef USERERROR
#define USERERROR(msg, id, ...) {					\
    char __buff[256];							\
    SNPRINTF(__buff, 255, ERROR_HDR msg MSG_NEWLINE, ##__VA_ARGS__);	\
    __buff[255] = 0;							\
    PRINTF_ERR(__buff, id);						\
  }
#undef USERWARNING
#define USERWARNING(msg, id, ...) {					\
    char __buff[256];							\
    SNPRINTF(__buff, 255, WARNING_HDR msg MSG_NEWLINE, ##__VA_ARGS__);	\
    __buff[255] = 0;							\
    PRINTF_WRN(__buff, id);						\
  }

#define HERE fprintf(stderr, "%s: %s(): %i: HERE\n", BASENAME(__FILE__), __FUNCTION__, __LINE__);

#ifndef DEBUG
#undef DMESSAGE
#define DMESSAGE(msg, level, ...)
#define TODO(msg, ...)
#define FENTER
#define FEXIT
#define DEBUG_STATISTICS
#else
#undef DMESSAGE
#define DMESSAGE(msg, level, ...) {					\
    if(level<=get_debug_mode()){					\
      char __buff[256];							\
      SNPRINTF(__buff, 255, " ** %s: %s(): %d: " msg "\n",		\
	       BASENAME(__FILE__), __FUNCTION__, __LINE__, ##__VA_ARGS__); \
      __buff[255] = 0;							\
      PRINTF_MSG(__buff);fflush(stdout);				\
    }									\
  }
#define TODO(msg, ...) {						\
    char __buff[256];							\
    SNPRINTF(__buff, 255, "TODO %s: %s(): %d: " msg "\n",		\
	     BASENAME(__FILE__), __FUNCTION__, __LINE__, ##__VA_ARGS__);\
    __buff[255] = 0;							\
    PRINTF_MSG(__buff);fflush(stdout);					\
  }


#define FENTER								\
  if(DEBUG_DETAILED<=get_debug_mode()){					\
    printf(" --------------------------\n");				\
    printf(" -- ENTER %s:%s\n", BASENAME(__FILE__), __FUNCTION__);	\
    printf(" --------------------------\n");				\
  }									\


#define FEXIT								\
  if(DEBUG_DETAILED<=get_debug_mode()){					\
    printf(" --------------------------\n");				\
    printf(" -- EXIT %s:%s\n", BASENAME(__FILE__), __FUNCTION__);	\
    printf(" --------------------------\n");				\
  }									\

#define DEBUG_STATISTICS				\
  {							\
    MESSAGE("Internal memory usage %s:%d bytes %lu",	\
	    __FILE__, __LINE__, get_total_memory_usage());	\
    print_allocated_pointers();				\
  }							\

#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

  int get_debug_mode(void);
  void set_debug_mode(int debug);

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif
