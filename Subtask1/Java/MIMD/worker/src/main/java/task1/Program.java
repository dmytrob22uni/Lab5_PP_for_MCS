package task1;

import mpi.*;

public class Program
{
    public static void main( String[] args ) throws MPIException
    {
        MPI.Init(args);

        Comm comm = MPI.COMM_WORLD;

        final int ROOT = 0;
        final int TAG_COUNT = 1;
        final int TAG_CHUNK = 2;
        final int TAG_RESULT = 3;
        final int TAG_TIME = 4;

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

        MPI.Finalize();
    }
}
