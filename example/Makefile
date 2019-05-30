CC=g++
CFLAGS=-Wall -pedantic -std=c++11

all: example1.out

example1.out: example1.cpp ${OBJS}
	${CC} ${CFLAGS} $< -I../eigen-git-mirror -I../src ../src/*.o -o $@
