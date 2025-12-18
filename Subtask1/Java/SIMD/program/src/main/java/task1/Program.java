package task1;

import mpi.*;

public class Program {
    public static void main(String[] args) throws MPIException {
        MPI.Init(args);
        Comm comm = MPI.COMM_WORLD;
        int rank = comm.getRank();
        int size = comm.getSize();

        // array size (can be passed via args)
        final int N =
            (args.length >= 1) ?
            Integer.parseInt(args[0]) :
            5_000_000;

        final int ROOT = 0;
        final int TAG_COUNT = 1;
        final int TAG_CHUNK = 2;
        final int TAG_RESULT = 3;
        final int TAG_TIME = 4;

        // compute smoothed elements distribution
        int baseAmount = N / size;
        int leftoversAmount  = N % size;
        int[] elementsPerChunk = new int[size];
        int[] startIndexPerChunk = new int[size];
        for (int i = 0; i < size; i++) {
            elementsPerChunk[i] =
                baseAmount +
                (i < leftoversAmount ? 1 : 0);
            startIndexPerChunk[i] =
                (i == 0) ?
                0 :
                startIndexPerChunk[i - 1] + elementsPerChunk[i - 1];
        }

        if (rank == ROOT) {
            // create data
            long[] data = new long[N];
            for (int i = 0; i < N; i++)
                data[i] = i % 1000;

            // calculate sequential sum
            long startSequential = System.nanoTime();
            long sumSequential = 0L;
            for (int i = 0; i < data.length; i++) {
                sumSequential += i % 1000;
            }
            long endSequential = System.nanoTime();
            float timeSequential = (endSequential - startSequential) / 1_000_000F;

            System.out.printf(
                "### sequential\nN: %d\ntotal sum: %d\ntime: %.3f ms\n\n",
                N,
                sumSequential,
                timeSequential);

            // send chunks subarrays
            for (int procIndex = 1; procIndex < size; procIndex++) {
//S//
                comm.send(
                    new int[]{elementsPerChunk[procIndex]},  // amount of elements
                    1,  // amount of data
                    MPI.INT,
                    procIndex,  // destination rank index
                    TAG_COUNT);  // message tag
                // send the chunk
                if (elementsPerChunk[procIndex] > 0) {
//S//
                    comm.send(
                        MPI.slice(data, startIndexPerChunk[procIndex]),  // start index
                        elementsPerChunk[procIndex],  // amount of data
                        MPI.LONG,
                        procIndex,  // destination rank index
                        TAG_CHUNK);  // message tag
                }
            }
            
            // sum calculation for root chunk
            long localSum = 0;
            int localCount = elementsPerChunk[ROOT];
            for (int i = startIndexPerChunk[ROOT]; i < startIndexPerChunk[ROOT] + localCount; i++)
                localSum += data[i];

            // receive chunk sums from workers and accumulate
            long total = 0L;
            total += localSum;
            for (int procIndex = 1; procIndex < size; procIndex++) {
                long[] chunkSum = new long[1];
//R//
                comm.recv(
                    chunkSum,  // where to put
                    1,  // amount of data
                    MPI.LONG,
                    procIndex,  // source rank index
                    TAG_RESULT);  // message tag
                total += chunkSum[0];
            }

            // receive chunk calculations pure time and accumulate
            float time = 0F;
            for (int procIndex = 1; procIndex < size; procIndex++) {
                float[] chunkTime = new float[1];
//R//
                    comm.recv(
                    chunkTime,  // where to put
                    1,  // amount of data
                    MPI.FLOAT,
                    procIndex,  // source rank index
                    TAG_TIME);  // message tag
                time += chunkTime[0];
            }

            System.out.printf(
                "### parallel\nN: %d\nprocesses: %d\ntotal sum: %d\npure time: %.3f ms\n",
                N,
                size,
                total,
                time);
        }
        else {
            // receive count
            int[] elementsInChunk = new int[1];
//R//
            comm.recv(
                elementsInChunk,  // where to put
                1,  // amount of data
                MPI.INT,
                ROOT,  // source rank index
                TAG_COUNT);  // message index


            int count = elementsInChunk[0];
            // length must be at least of size one, to at least contain "0" element,
            // so in later "send" method non-zero-length array will be passed
            long[] chunk = new long[Math.max(1, count)];
            // receive data
            if (count > 0) {
//R//
                comm.recv(
                    chunk,  // where to put
                    count,  // amount of data
                    MPI.LONG,
                    ROOT,  // source rank index
                    TAG_CHUNK);  // message tag
            }
            // compute chunk sum
            long startTime = System.nanoTime();
            long localSum = 0;
            for (int i = 0; i < count; i++)
                localSum += chunk[i];
            long endTime = System.nanoTime();
            float localTime = (endTime - startTime) / 1_000_000F;
            // send chunk sum back
//S//
            comm.send(
                new long[]{localSum},  // chunk sum
                1,  // amount of data
                MPI.LONG,
                ROOT,  // destination rank index
                TAG_RESULT);  // message tag
            // send chunk calculations time back
//S//
            comm.send(
                new float[]{localTime},  // chunk calculations time
                1,  // amount of data
                MPI.FLOAT,
                ROOT,  // destination rank index
                TAG_TIME);  // message tag
        }

        MPI.Finalize();
    }
}

