all:
	gcc -Wall tests.c -lm -o tests.o

test:
	./tests.o

model:
	gcc -Wall model.c -lm -o model.o

clean:
	rm -f *.o
