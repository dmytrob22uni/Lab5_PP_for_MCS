# -O3 - 3rd level of g++ optimizations, -std=c++17 - this level of cpp standard
mpicxx -O3 -std=c++17 mpi_pair_sum.cpp -o mpi_pair_sum
# run with 8 processors
mpirun -np 8 ./mpi_pair_sum

