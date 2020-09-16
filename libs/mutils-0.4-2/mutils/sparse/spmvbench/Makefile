LIB=-lm

CC=gcc

OPTFLAGS=-DMPICH_IGNORE_CXX_SEEK

INCNUMA=-I$(HOME)/libs/numactl-0.9.9/
LIBNUMA=-L/usr/lib64/ /usr/lib64/libnuma.so.1

METIS_HOME=$(HOME)/work/projects/metis
METIS_INC=-I$(METIS_HOME)/Lib
METIS_LIB=-L$(METIS_HOME) -lmetis

LIBUTILS_HOME=../../libutils

DEBUG=
OPTFLAGS=-Wall -O3 -funroll-all-loops

# -funroll-all-loops -Wall
# -fpeephole2 -fschedule-insns2 -ftree-vectorize -ftree-vectorizer-verbose=0 

FLAGS=-Wall -I../../ -Ircm $(DEBUG) $(OPTLFAGS) $(METIS_INC) -fopenmp -DUSE_METIS
#FLAGS=-Wall -I../../ -Ircm -funroll-all-loops $(DEBUG) $(OPTLFAGS) $(METIS_INC) -fopenmp -DUSE_METIS

LDFLAGS=-L$(LIBUTILS_HOME) -lutils -lpthread -lm $(DEBUG)  $(METIS_LIB) -fopenmp
LDOMP=-lpthread

SRC = main.c sp_matv.c sparse.c matrix_import.c distribute.c comm.c reorder.c reorder_metis.c reorder_random.c reorder_rcm.c thread_routine.c
OBJ = main.o sp_matv.o sparse.o matrix_import.o distribute.o comm.o reorder.o reorder_metis.o reorder_random.o reorder_rcm.o thread_routine.o

.c.o: %.c %.h
	$(CC) $(INC) $(FLAGS) $(OPTFLAGS) -c $<

all: $(OBJ) libutils
	$(CC) $(OBJ) -o main $(LDFLAGS)

libutils:
	#cd $(LIBUTILS_HOME) && use_openmp=1 ./configure.sh
	make -C	$(LIBUTILS_HOME) clean
	make -C	$(LIBUTILS_HOME)

clean:
	rm $(OBJ) $(PROGRAM) *o
	cd mlib && make clean
