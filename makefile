all:
	nvcc -Xcompiler -fopenmp -O3 main.cu -o prog
clean:
	rm prog