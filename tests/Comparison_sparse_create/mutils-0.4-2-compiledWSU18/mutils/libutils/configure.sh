#!/bin/bash

: ${use_numa=0}
: ${use_openmp=0}
: ${use_dmalloc=0}
: ${cc=gcc}

#cp config.h.template config.h
rm Makefile.inc

echo '.c.o:	%.c %.h'  >> Makefile.inc
printf "\t"  >> Makefile.inc
echo '$(CC) $(INCLUDES) $(CFLAGS) $(OPTFLAGS) -c $<'  >> Makefile.inc
echo 'COMPILE = $(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDES) $(LIBS)'  >> Makefile.inc

compiler=`basename $cc`

if [[ $compiler == "mex" ]]; then
    echo "CC=$cc" >> Makefile.inc
    if [[ $use_openmp -eq 1 ]]; then
	ompflags="-fopenmp"
    fi
    echo "OPTFLAGS += -largeArrayDims -O COPTIMFLAGS='-std=gnu89 -O3 -DNDEBUG -Wall $ompflags'" >> Makefile.inc
fi

if [[ $compiler == "gcc" ]]; then
    echo "CC=$cc" >> Makefile.inc
    echo "CFLAGS = -ftree-vectorize -std=c99 -O3 -Wall -ftree-vectorizer-verbose=0 -funroll-all-loops -D_GNU_SOURCE" >> Makefile.inc
    echo "OPTFLAGS += -fPIC" >> Makefile.inc
    ompflags="-fopenmp"
fi

if [[ $compiler == "icc" ]]; then
    echo "CC=$cc"  >> Makefile.inc
    echo "CFLAGS = -fno-alias -xW -O3 -vec-report=1"  >> Makefile.inc
    echo "OPTFLAGS += -fPIC" >> Makefile.inc
    ompflags="-openmp"
fi

if [[ $use_numa -eq 1 ]]; then
    echo "#define USE_NUMA 1" >> config.h
    echo "
NUMA_HOME = \$(HOME)/work/projects/libs/
OPTFLAGS += -I\$(NUMA_HOME)/include
OPTLIBS  += -Wl,-rpath -Wl,\$(NUMA_HOME)/lib64 -L\$(NUMA_HOME)/lib64 -lnuma
" >> Makefile.inc
fi

if [[ $use_dmalloc -eq 1 ]]; then
    echo "#define USE_DMALLOC 1" >> config.h
    echo "
DMALLOC_HOME = \$(HOME)/work/projects/libs/
OPTFLAGS += -I\$(DMALLOC_HOME)/include -DDMALLOC -DDMALLOC_FUNC_CHECK
OPTLIBS  += -L\$(DMALLOC_HOME)/lib -ldmallocth
" >> Makefile.inc
fi

if [[ $use_openmp -eq 1 ]]; then
    echo "#define USE_OPENMP 1" >> config.h
    echo "USE_OPENMP = 1" >> Makefile.inc
    if [[ $cc != "mex" ]]; then
	echo "OPTFLAGS += $ompflags"  >> Makefile.inc
    fi
fi
