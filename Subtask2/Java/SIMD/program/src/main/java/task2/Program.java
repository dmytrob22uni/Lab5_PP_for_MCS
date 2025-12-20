package task2;

import mpi.*;

public class Program {

    static final int TASK_TAG = 1;
    static final int RESULT_TAG = 2;
    static final long TERMINATE_INDEX = -1L;

    public static void main(String[] args) throws Exception {

        MPI.Init(args);

        Comm comm = MPI.COMM_WORLD;

        int rank = comm.getRank();
        int size = comm.getSize();

        final int N = 5_000_000;

        if (rank == 0) {
            calculateSequentialSum(N);
        }

        if (size == 1) {
            MPI.Finalize();
            return;
        }

        if (rank == 0) {  // MASTER

            long[] data = new long[N];
            for (int i = 0; i < N; i++)
                data[i] = i % 1000L;

            long t0 = System.nanoTime();

            int loopLength = N;
            while (loopLength > 1) {

                int halvedLoopLength = (loopLength + 1) / 2;
                int taskIndex = 0;

                // prewarm: send up to "size - 1" jobs to workers to fill them all till the next while-"job-or-termination"-loop
                for (int processRank = 1; processRank < size && taskIndex < halvedLoopLength; processRank++) {

                    sendTaskToWorker(
                        comm,                                  // communicator
                        processRank,                           // destination process rank
                        taskIndex,                             // task index
                        data[taskIndex],                       // left element
                        taskIndex ==
                            (loopLength - 1 - taskIndex) ?
                            0L :
                            data[loopLength - 1 - taskIndex],  // right element
                        taskIndex ==
                            (loopLength - 1 - taskIndex) ?
                            1L :
                            0L);                               // "is same element" flag

                    taskIndex++;
                }

                int receivedPairSumsAmount = 0;
                // buffer for receiving calculated pair sum: [leftIndex, sum]
                long[] resBuf = new long[2];
                Status status;

                while (receivedPairSumsAmount < halvedLoopLength) {

                    status = comm.recv(
                        resBuf,          // where to put sum
                        2,               // capacity of resBuf
                        MPI.LONG,
                        MPI.ANY_SOURCE,  // rank of source
                        RESULT_TAG);     // message tag

                    int workerRank = status.getSource();  // worker, which received task
                    int leftIndex = (int) resBuf[0];      // left index of pair sum
                    long sum = resBuf[1];                 // actual sum

                    data[leftIndex] = sum;
                    receivedPairSumsAmount++;

                    // if tasks remain after prewarm phase + already given ones, give the worker a new task immediately
                    if (taskIndex < halvedLoopLength) {

                        sendTaskToWorker(
                            comm,                                  // communicator
                            workerRank,                            // destination process rank
                            taskIndex,                             // next task index
                            data[taskIndex],                       // left element
                            taskIndex ==
                                (loopLength - 1 - taskIndex) ?
                                0L :
                                data[loopLength - 1 - taskIndex],  // right element
                            taskIndex ==
                                (loopLength - 1 - taskIndex) ?
                                1L :
                                0L);                               // "is same element" flag

                        taskIndex++;
                    }
                    // else {
                    //     // no tasks remain to be passed, next iteration on receiving gonna be last
                    // }
                }

                // shirk active length on complete iteration over half of the array
                loopLength = halvedLoopLength;
            }

            long t1 = System.nanoTime();

            System.out.println("sum (MPI): " + data[0]);
            System.out.printf("time: %.3f ms\n", (t1 - t0) / 1_000_000F);

            // tell workers to terminate
            for (int processRank = 1; processRank < size; processRank++) {

                // buffer with termination order inside
                long[] termBuf = new long[] { TERMINATE_INDEX, 0L, 0L, 0L };
                comm.send(
                    termBuf,      // what to send
                    4,            // capacity
                    MPI.LONG,
                    processRank,  // rank of destination
                    TASK_TAG);    // message tag
            }

        } else {  // WORKER

            while (true) {

                // buffer for receiving task of pair sum calculation: [workerRank, leftElement, rightElement, isSameFlag]
                long[] taskBuf = new long[4];
                comm.recv(
                    taskBuf,    // where to put
                    4,          // capacity
                    MPI.LONG,
                    0,          // rank of source
                    TASK_TAG);  // message tag
                
                // check whether master sent termination order
                long workerRankLong = taskBuf[0];
                if (workerRankLong == TERMINATE_INDEX) {
                    break;
                }

                int workerRank = (int) workerRankLong;
                long leftElement = taskBuf[1];
                long rightElement = taskBuf[2];
                long isSameFlag = taskBuf[3];

                long sum =
                    (isSameFlag == 1L) ?
                    leftElement :
                    leftElement + rightElement;

                // buffer for sending calculated pair sum: [leftIndex, sum]
                long[] resBuf = new long[] { workerRank, sum };
                comm.send(
                    resBuf,       // what to send
                    2,            // capacity
                    MPI.LONG,     
                    0,            // rank of source
                    RESULT_TAG);  // message tag

                // loop to wait for next task / termination order
            }
        }

        MPI.Finalize();
    }

    // helper function to send a task to a worker (workerRank, leftElement, rightElement, isSameFlag)
    static void sendTaskToWorker(
        Comm comm,
        int workerRank,
        int taskIndex,
        long leftElement,
        long rightElement,
        long isSameFlag) throws MPIException {

        long[] buf = new long[4];
        buf[0] = taskIndex;
        buf[1] = leftElement;
        buf[2] = rightElement;
        buf[3] = isSameFlag;
        comm.send(
            buf,         // what to send
            4,           // capacity
            MPI.LONG,
            workerRank,  // rank of destination
            TASK_TAG);   // message tag
    }

    static void calculateSequentialSum(int N) {

        long[] data = new long[N];
        for (int i = 0; i < N; i++)
            data[i] = i % 1000L;

        long t0 = System.nanoTime();

        int loopLength = N;
        while (loopLength > 1) {

            int halvedLoopLength = (loopLength + 1) / 2;
            for (int indexLeft = 0; indexLeft < halvedLoopLength; indexLeft++) {

                int indexRight =
                    loopLength - 1 -
                    indexLeft;

                if (indexLeft != indexRight) {
                    data[indexLeft] =
                        data[indexLeft] +
                        data[indexRight];
                }
            }
            loopLength = halvedLoopLength;
        }

        long t1 = System.nanoTime();

        System.out.println("sum (sequential): " + data[0]);
        System.out.printf("time: %.3f ms\n", (t1 - t0) / 1_000_000F);
        System.out.println();
    }
}
