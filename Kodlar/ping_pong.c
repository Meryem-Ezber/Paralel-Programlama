#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI kutuphanesi

#define N 10
#define NADAL 0
#define FEDERER 1
#define TAG 666

int main(void) {

int rank, size;

MPI_Init(NULL, NULL);

MPI_Status status;

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

if (size != 2)
    MPI_Abort(MPI_COMM_WORLD, 0);

// initialization
int a, i;
if (rank == NADAL)
    a = 2;

// synchronization
MPI_Barrier(MPI_COMM_WORLD);

// messaging part
for (i = 0; i < N; i++) {

if (rank == NADAL) {
    MPI_Send(&a, 1, MPI_INT, FEDERER, TAG, MPI_COMM_WORLD);
    MPI_Recv(&a, 1, MPI_INT, FEDERER, TAG+1, MPI_COMM_WORLD, &status);
    a++;
} else {
    // MPI_Send(&a, 1, MPI_INT, NADAL, TAG+1, MPI_COMM_WORLD); order of send and recv routines is crucial
    MPI_Recv(&a, 1, MPI_INT, NADAL, TAG, MPI_COMM_WORLD, &status);
    a++;
    MPI_Send(&a, 1, MPI_INT, NADAL, TAG+1, MPI_COMM_WORLD);
}

}

if (rank == NADAL)
    printf("a = %d\n", a);

MPI_Finalize();

}






