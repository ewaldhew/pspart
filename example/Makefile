LIBTOOL=libtool --mode=link
CC=gcc
CXX=g++
CFLAGS=-Wall -pedantic -Og -DDEBUG

INCLUDES = -I../src ../src/libpspart.la

all: example1.out example2.out

example1.out: example1.cpp
	${LIBTOOL} ${CXX} ${CFLAGS} -std=c++11 $< ${INCLUDES} -I../eigen-git-mirror -o $@

example2.out: example2.c
	${LIBTOOL} ${CC} ${CFLAGS} -std=c99 $< ${INCLUDES} -o $@
