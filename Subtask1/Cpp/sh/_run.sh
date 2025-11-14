mpic++ -O3 -std=c++17 -o _sum_mpi _sum_mpi.cpp
mpirun -np 4 ./_sum_mpi

