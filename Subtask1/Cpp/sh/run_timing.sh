# -O3 - maximum optimization; -O - g++ optimization flag
# -std - assumes that input .cpp files are for that standard
mpic++ -O3 -std=c++17 -o sum_mpi_timing sum_mpi_timing.cpp
# number of processes to run each instance of program on
mpirun -n 10 ./sum_mpi_timing

