#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI kutuphanesi

#define MASTER 0

int main(void) {

int rank, size, a = 0;

MPI_Init(NULL, NULL);

MPI_Status status;

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

printf("rank = %d --> a = %d\n", rank, a);
MPI_Barrier(MPI_COMM_WORLD);

if (rank == MASTER)
     a = 8;

printf("rank = %d --> a = %d\n", rank, a);
MPI_Barrier(MPI_COMM_WORLD);

MPI_Bcast(&a, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
printf("rank = %d --> a = %d\n", rank, a);

MPI_Finalize();

return 0;

}
