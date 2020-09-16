PROGNAME=interp_test

MATHFLAGS=-mavx

all: test.c
	make -C ../libutils
	gcc -g -DONE_BASED_INDEX=1 -ansi -fexceptions -fPIC -fno-omit-frame-pointer -pthread -std=c99 -D_GNU_SOURCE -Wall -DUSE_OPENMP -fopenmp -funroll-loops -finline-functions -O2 -DNDEBUG -I../ $(MATHFLAGS) test.c einterp_tri.c einterp_quad.c ../libutils/libutils.a -o $(PROGNAME)

clean:
	make -C ../libutils clean
	rm *.o
	rm $(PROGNAME)
