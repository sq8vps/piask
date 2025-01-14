plain:
	gcc -o piask main.c plain.c my_timers.c -lm -DUSE_PLAIN

openmp:
	gcc -fopenmp -o piask main.c openmp.c my_timers.c -lm -DUSE_OPENMP

mpi:
	mpicc -o piask main.c mpi.c my_timers.c -lm -DUSE_MPI

cuda:
	nvcc -o piask main.c cuda.cu my_timers.c -lm -DUSE_CUDA

