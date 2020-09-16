include Makefile.inc

OBJECTS  = $(patsubst %.c,%.o,$(wildcard *.c))
SOURCES  = %.c %h
LIBNAME = libutils.a


all: 	$(LIBNAME)

$(LIBNAME):	$(OBJECTS)
	ar r $(LIBNAME) $(OBJECTS)

clean:
	rm -f $(OBJECTS) $(LIBNAME)
