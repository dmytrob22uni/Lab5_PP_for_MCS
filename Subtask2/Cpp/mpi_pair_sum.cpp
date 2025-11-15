#include <mpi.h>
#include <chrono>
#include <iostream>
#include <vector>

using int64 = long long;

std::vector<int64> generate_array(std::size_t N) {
    std::vector<int64> array(N);
    for (std::size_t index = 0; index < N; ++index) {
        array[index] = static_cast<int64>(index % 1000);
    }
    return array;
}

// sequential pairwise reduction: a[i] += a[length-1-i]
// keep first ceil(length / 2) items
int64 sequential_pairwise(std::vector<int64> array) {
    std::size_t N = array.size();
    while (N > 1) {
        std::size_t newN = (N + 1) / 2;
        for (std::size_t index = 0; index < N / 2; ++index) {
            array[index] = array[index] + array[N - 1 - index];
        }
        N = newN;
    }
    return array[0];
}

// compute counts / offsets
void make_counts_offsets(
    int newN,
    int processes,
    std::vector<int>& counts,
    std::vector<int>& offsets)
{
    // arrays of '0'-s of procs length
    counts.assign(processes, 0);
    offsets.assign(processes, 0);

    int base = newN / processes;
    int rem = newN % processes;

    for (int rankIndex = 0; rankIndex < processes; ++rankIndex) {
        counts[rankIndex] = base + (rankIndex < rem ? 1 : 0);
    }
    offsets[0] = 0;
    for (int rankIndex = 1; rankIndex < processes; ++rankIndex) {
        offsets[rankIndex] = offsets[rankIndex-1] + counts[rankIndex-1];
    }
}

// MPI parallel pairwise reduction
// root (rank 0) must provide array of length N
// returns the final value (sum) only valid on root
// time is measured on root.
int64 mpi_pairwise(
    std::vector<int64>& array,
    MPI_Comm communicator,  // defines which MPI processes participate in this function and how they communicate with each other
    double &elapsed_seconds)
{
    int rank, processes;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &processes);

    const int N = static_cast<int>(array.size());
    std::vector<int64> sums_local; // will hold computed partial outputs for each iteration
    std::vector<int> counts, offsets;

    // allocate buffer on every rank with capacity N
    std::vector<int64> array_buffer(N);

    // copy initial array into buf on root; others don't need init but keep capacity
    if (rank == 0) {
        array_buffer = array;
    }

    // ensures all ranks start the timed code at the same moment
    MPI_Barrier(communicator);

    double t0 = 0.0;
    if (rank == 0) {
        // only rank 0 records the time because it is the one measuring performance
        t0 = MPI_Wtime();
    }

    int length = N;
    while (length > 1) {
        // broadcast first length elements to everyone
        MPI_Bcast(
            array_buffer.data(),
            length,
            MPI_LONG_LONG,
            0,
            communicator);

        int newN = (length + 1) / 2;
        make_counts_offsets(
            newN,
            processes,
            counts,
            offsets);
        int count_local = counts[rank];  // for each rank particular amount of elements to sum
        sums_local.assign(count_local, 0);  // array of '0'-s of count_local length

        // compute local segment: indices [offsets[rank], offsets[rank]+local_count-1]
        for (int index = 0; index < count_local; ++index) {
            int index_global = offsets[rank] + index;
            int index_global_opposite = length - 1 - index_global;
            if (index_global == index_global_opposite) {
                // set middle element when length is odd
                sums_local[index] = array_buffer[index_global];
            } else {
                // make regular summation
                sums_local[index] =
                    array_buffer[index_global] +
                    array_buffer[index_global_opposite];
            }
        }

        // prepare arrays copies for MPI_Gatherv on root
        // std::vector<int> counts_copy(processes);
        // std::vector<int> offsets_copy(processes);
        // for (int rankIndex = 0; rankIndex < processes; ++rankIndex) {
        //     counts_copy[rankIndex] = counts[rankIndex];
        //     offsets_copy[rankIndex] = offsets[rankIndex];
        // }

        // take the partial results from all MPI processes and assemble them into one final array on rank 0 only
        MPI_Gatherv(
            sums_local.data(),
            count_local,
            MPI_LONG_LONG,
            array_buffer.data(),
            counts.data(),
            offsets.data(),
            MPI_LONG_LONG,
            0,
            communicator);

        // updates length for next iteration on root
        length = newN;
    }

    // ensure the computations is finished on all ranks before the timing is stopped
    MPI_Barrier(communicator);

    // only rank 0 computes the end timestamp
    // only rank 0 has the complete reduced array after MPI_Gatherv
    if (rank == 0) {
        double t1 = MPI_Wtime();
        elapsed_seconds = t1 - t0;
        return array_buffer[0];
    // non-rank-0 return dummy values
    } else {
        elapsed_seconds = 0.0;
        return 0;
    }
}

int main(int argc, char** argv) {
    const std::size_t N = 2'000'000;

    // initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm communicator = MPI_COMM_WORLD;
    int rank, processes;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &processes);

    if (rank == 0) {
        std::cout << "n = " << N << ", processes = " << processes << "\n";
    }

    // create and broadcast array form rank 0 to all the rest ranks
    std::vector<int64> array;
    if (rank == 0) {
        array = generate_array(N);
    } else {
        array.resize(N);  // reserve valid memory buffer
    }
    MPI_Bcast(
        array.data(),
        N,
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD);

    // run sequential on root only and measure time
    double time_sequential = 0.0;
    int64 sum_sequential = 0;
    if (rank == 0) {
        auto start = std::chrono::steady_clock::now();
        sum_sequential = sequential_pairwise(array);
        auto end = std::chrono::steady_clock::now();
        time_sequential = std::chrono::duration<double>(end - start).count();
        std::cout << "sum sequential = " << sum_sequential << ", time = " << time_sequential << " s\n";
    }

    // run MPI parallel reduction
    double time_mpi = 0.0;
    int64 sum_mpi = mpi_pairwise(array, communicator, time_mpi);

    if (rank == 0) {
        std::cout << "sum MPI = " << sum_mpi << ", time = " << time_mpi << " s\n";
        std::cout << (
            sum_sequential == sum_mpi ?
            "RESULTS MATCH! Very gud!\n" :
            "RESULTS DIFFER! Disgusting! Your grade is -50, get out!\n");
    }

    // clean up internal resources like communicators, buffers, and message queues
    MPI_Finalize();
    return 0;
}

