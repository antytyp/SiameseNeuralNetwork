all:
	gcc -Wall -lm tests.c -o tests.o

test:
	./tests.o

clean:
	rm -f *.o
