#include <mpi.h>  // main MPI header
#include <vector>  // dynamic arrays
#include <numeric>  // for std::accumulate
#include <iostream>  // IO
#include <cstdint>  // fixed-width types (int64_t)
#include <cassert>  // simple runtime checks
#include <chrono>  // time measurements

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // MPI initialization

    // get number of processes and current process rank
    int world_size = 0, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // number of MPI processed lauched (mpirun -np X)
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  // ID of this process

    const std::int64_t n = 2000000;

    std::vector<int> numbers_per_rank(world_size), offsets(world_size);  // numbers_per_rank vector for elements distribution, displs vector for starting positions (offsets)
    std::int64_t base = n / world_size;  // base of division among MPI processes
    int rem = (int)(n % world_size);  // remainders to put 1 to each MPI proces till all are distributed
    for (int r = 0; r < world_size; ++r) {
        numbers_per_rank[r] = (int)base + (r < rem ? 1 : 0);
    }
    offsets[0] = 0;
    for (int r = 1; r < world_size; ++r) {
        offsets[r] = offsets[r-1] + numbers_per_rank[r-1];
    }

    std::vector<long long> local_buf(numbers_per_rank[world_rank]); // each process allocates space exactly for its own chunk

    std::vector<long long> full_data;
    long long sum_single = 0;

    if (world_rank == 0) {  // only generate on root rank
        full_data.resize((size_t)n);  // unsigned long type guaranteed to be able to index any object in memory

        for (std::int64_t i = 0; i < n; ++i) {
            full_data[(size_t)i] = i % 1000;
        }
        // compute reference sum to check results
        sum_single = std::accumulate(
            full_data.begin(),  // first
            full_data.end(),  // last
            0LL);  // init
    }

    // Scatter|v| supports variable sizes of chunks
    // rank 0 sends appropriae chunk to each rank
    MPI_Scatterv(
        world_rank == 0 ? full_data.data() : nullptr, // sendbuf (only meaningful at root)
        numbers_per_rank.data(),  // sendcounts
        offsets.data(),  // displacements
        MPI_LONG_LONG,  // send type
        local_buf.data(),  // recvbuf
        numbers_per_rank[world_rank],  // recvcount
        MPI_LONG_LONG,  // recvtype
        0,  // root
        MPI_COMM_WORLD  // mpi communicator
    );

    // each rank computes local sum - done independently across all ranks
    long long local_sum = 0;
    for (long long v : local_buf) {
        local_sum += v; 
    }

    // reduce local sums to global sum at root
    // no rank can pass this point until all have participated
    long long sum_parallel = 0;
    MPI_Reduce(
        &local_sum,
        &sum_parallel,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    if (world_rank == 0) {
        std::cout << "n = " << n << ", ranks = " << world_size << "\n";
        std::cout << "total sum single (root local compute) = " << sum_single << "\n";
        std::cout << "total sum parallel (MPI_Reduce) = " << sum_parallel << "\n";
        std::cout << (sum_single == sum_parallel ? "OK: sums match\n" : "ERROR: sums mismatch! PANIC! OMG! WTH!\n");
    }
    
    MPI_Finalize();  // releases MPI internal resources.
    return 0;  // return code for "ok" run
}

