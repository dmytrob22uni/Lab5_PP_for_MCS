#include <mpi.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <iomanip>

// single threaded sequential sum
long long single_sum(
    const std::vector<long long>& data,  // input data (by ref)
    double &time)  // time elapsed (by ref)
{
    using clock = std::chrono::high_resolution_clock;
    auto time_start = clock::now();
    long long sum = std::accumulate(
        data.begin(),
        data.end(),
        0LL);
    auto time_end = clock::now();
    time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_end - time_start).count();
    return sum;
}

// MPI scatter + local sum + reduce (returned sum valid on root only)
long long mpi_sum(
    const std::vector<long long>* data_ptr,  // pointer to full data on root (nullptr on non-root ranks)
    std::vector<long long>& data_slice,  // buffer on each rank that will receive its chunk
    const std::vector<int>& chunks,  // counts arrays (same on all ranks)
    const std::vector<int>& offsets,  // displacements arrays (same on all ranks)
    int world_rank,  // MPI context
    int world_size,  // MPI context
    double &root_time_total,  // total elapsed time measured on this rank (root prints its value)
    double &all_time_local,  // maximum local compute time among ranks (root prints)
    double &all_time_total)  // maximum total time among ranks (root prints)
{
    using clock = std::chrono::high_resolution_clock;

    // synchronize all ranks so timing is comparable
    MPI_Barrier(MPI_COMM_WORLD);

    auto time_start_total = clock::now();

    // Scatterv - distribute parts of data (only meaningful at root)
    MPI_Scatterv(
        // only root process have the data in memory, non-root ones do not need it, so pass nullptr
        (world_rank == 0 && data_ptr != nullptr) ? data_ptr->data() : nullptr,
        const_cast<int*>(chunks.data()),
        const_cast<int*>(offsets.data()),
        MPI_LONG_LONG,
        data_slice.data(),
        chunks[world_rank],
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD
    );

    // measure local compute time (sum of local_buf on each rank)
    auto time_start_local = clock::now();
    long long sum_local = 0;
    for (long long v : data_slice) {
        sum_local += v;
    }
    auto time_end_local = clock::now();
    double time_local = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_end_local - time_start_local).count();

    // reduce local sums to root
    // in other words - combine values from all ranks
    long long sum_global = 0;
    MPI_Reduce(
        &sum_local,
        &sum_global,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        0,
        MPI_COMM_WORLD);

    auto time_end_total = clock::now();
    double time_total = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_end_total - time_start_total).count();

    // get combination of local_sum compute times across ranks 
    MPI_Reduce(
        &time_local,
        &all_time_local,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD);
    // get combination of local_sum + reduce_stage compute times across ranks 
    MPI_Reduce(
        &time_total,
        &all_time_total,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD);

    // root's total time is the same as total time of all ranks (local_sum + reduce_stage)
    if (world_rank == 0) {
        root_time_total = time_total;
    }

    return sum_global;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size = 0, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // multiprocessing proof
    char name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(name, &name_len);
    std::cout << "rank #" << world_rank 
        << " of " << world_size 
        << " on " << name << "\n";

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "\n";
    }

    const std::int64_t n = 2000000; // array length

    // prepare counts and displacements for Scatterv
    // (chunks and offsets accordingly)
    std::vector<int> chunks(world_size), offsets(world_size);
    std::int64_t base = n / world_size;
    int rem = static_cast<int>(n % world_size);
    for (int rank_index = 0; rank_index < world_size; ++rank_index) {
        // add +1 of remainders to somehow equalize rank chunks
        chunks[rank_index] =
            static_cast<int>(base + (rank_index < rem ? 1 : 0));
    }
    offsets[0] = 0;
    for (int rank_index = 1; rank_index < world_size; ++rank_index) {
        // calculate offset for each rank chunk
        offsets[rank_index] =
            offsets[rank_index-1] + chunks[rank_index-1];
    }

    // each rank allocates local buffer of the appropriate size
    std::vector<long long> local_buf(chunks[world_rank]);

    // root creates data and computes the single-threaded reference sum & time
    std::vector<long long> data;
    long long sum_single = 0;
    double time_single = 0.0;

    if (world_rank == 0) {
        data.resize(static_cast<size_t>(n));
        for (std::int64_t index = 0; index < n; ++index) {
            data[static_cast<size_t>(index)] = index % 1000;
        }

        sum_single = single_sum(data, time_single);
    }

    // MPI scatter + local sum + reduce and measure timings
    double root_time_total = 0.0;
    double all_time_local = 0.0;
    double all_time_total = 0.0;
    long long sum_parallel = mpi_sum(
        (world_rank == 0 ? &data : nullptr),
        local_buf,
        chunks,
        offsets,
        world_rank,
        world_size,
        root_time_total,
        all_time_local,
        all_time_total
    );

    if (world_rank == 0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "n = " << n << ", ranks = " << world_size << "\n";
        std::cout << "\n";

        std::cout << "Single threaded (root) sum = " << sum_single << "\n";
        std::cout << "Single threaded elapsed time = " << time_single << " ms\n";
        std::cout << "\n";

        std::cout << "Parallel (MPI) sum = " << sum_parallel << "\n";
        std::cout << "All per-rank local_sum elapsed time = " << all_time_local << " ms\n";
        std::cout << "All per-rank local_sum + reduce_stage elapsed time = " << all_time_total << " ms\n";
        std::cout << "\n";

        std::cout <<
            (sum_single == sum_parallel ?
                "OK: sums match\n" :
                "ERROR: sums mismatch!\n");
    }

    MPI_Finalize();
    return 0;
}

