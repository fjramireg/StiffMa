/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef _OSYSTEM_H
#define _OSYSTEM_H

#if defined(_WIN32)
#  define WINDOWS
#endif

#ifdef WINDOWS
#  define BASENAME
#else
#  define BASENAME basename
#endif

#ifdef _MSC_VER
/* 'function' : unreferenced inline function has been removed */
#  pragma warning(disable : 4514)
/* 'bytes' bytes padding added after construct 'member_name' */
#  pragma warning(disable : 4820)
/* turn on sse2 on Visual Studio */
#  define __SSE2__
#  ifdef _STDC_
#    define __STRICT_ANSI__
#  endif
#endif /* _MSC_VER */

#ifdef __APPLE__
#define APPLE
#endif

#ifdef __ICC
/* external function definition with no prior declaration */
#pragma warning disable 1418
/* external declaration in primary source file */
#pragma warning disable 1419
/* floating-point equality and inequality comparisons are unreliable */
#pragma warning disable 1572
/*  #pragma once is obsolete. Use #ifndef guard instead. */
#pragma warning disable 1782
#endif

#ifdef __GNUC__
#define GCC_VERSION (__GNUC__ * 10000			\
		     + __GNUC_MINOR__ * 100		\
		     + __GNUC_PATCHLEVEL__)
#else
#define GCC_VERSION 0
#endif

#ifndef __STRICT_ANSI__
#  define STATIC static
#  define HAVE_INLINE 1
#  ifdef _MSC_VER
#    define INLINE __inline
#  elif defined __GNUC__
#    define INLINE inline
#  else
#    define HAVE_INLINE 0
COMPILE_MESSAGE("No inline functions - unknown compiler")
#  endif
#else
#  ifdef __GNUC__
#    define HAVE_INLINE 1
/* in GCC this works despite the ANSI mode */
#    define INLINE __inline__
#    define STATIC static
#  else
COMPILE_MESSAGE("No inline functions.")
#    define HAVE_INLINE 0
#    define INLINE
#    define STATIC
#  endif
#endif /* __STRICT_ANSI__ */

#endif /* _OSYSTEM_H */
