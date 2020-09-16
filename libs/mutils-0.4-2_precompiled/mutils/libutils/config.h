#include "osystem.h"

#ifdef _MSC_VER
#define SNPRINTF _snprintf
#else
#define SNPRINTF snprintf
#endif

#ifdef  _GNU_SOURCE
# define __USE_GNU      1
# define __USE_XOPEN2K8 1
#endif

