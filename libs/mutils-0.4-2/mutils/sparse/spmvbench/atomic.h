#ifndef _ATOMIC_H
#define _ATOMIC_H

/*
  Code taken fromthe Linux kernel
 */

typedef struct { int counter; } atomic_t;

# define _ASM_ALIGN     " .balign 4 "
# define _ASM_PTR       " .quad "
#define LOCK_PREFIX \
                ".section .smp_locks,\"a\"\n"   \
                _ASM_ALIGN "\n"                 \
                _ASM_PTR "661f\n" /* address */ \
                ".previous\n"                   \
                "661:\n\tlock; "
 
static __inline__ void atomic_add(int i, atomic_t *v){
  __asm__ __volatile__(
		       LOCK_PREFIX "addl %1,%0"
		       :"=m" (v->counter)
		       :"ir" (i), "m" (v->counter));
}

static __inline__ int atomic_add_return(int i, atomic_t *v)
{
  int __i = i;
  __asm__ __volatile__(
		       LOCK_PREFIX "xaddl %0, %1"
		       :"+r" (i), "+m" (v->counter)
		       : : "memory");
  return i + __i;
}
 
static __inline__ void atomic_inc(atomic_t *v)
{
  __asm__ __volatile__(
		       LOCK_PREFIX "incl %0"
		       :"=m" (v->counter)
		       :"m" (v->counter));
}


static __inline__ void atomic_init(atomic_t *v, int i)
{
  v->counter = i;
}
#endif /* _ATOMIC_H */
