all:
	gcc -Wall tests.c -lm -o tests.o

test:
	./tests.o

simple_model:
	gcc -Wall model.c -lm -o model.o
    
advanced_model:
	gcc -Wall advanced_model.c -lm -o advanced_model.o

clean:
	rm -f *.o
