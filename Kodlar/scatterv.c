#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI kutuphanesi

#define MASTER 0

int *create1DArray(int n) {
     int *T = (int *)malloc(n*sizeof(int));
     return T;
}

void printArray(int *d, int n) {
int i;
for (i = 0; i < n; i++)
     printf("%d\t", d[i]);
puts("");
}


int main(void) {

int rank, size, i, j;

MPI_Init(NULL, NULL);

MPI_Status status;

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int *d; 

if (rank == MASTER) {
    d = create1DArray(12);     
    for (i = 0; i < 12; i++)
          d[i] = i+1;
}

int counts[4] = {2, 4, 2, 3};
int displs[4] = {0, 2, 7, 9}; 

int *local = create1DArray(counts[rank]);
for (i = 0; i < counts[rank]; i++)
     local[i] = 0;

printArray(local, counts[rank]);

MPI_Barrier(MPI_COMM_WORLD);

/*
int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype,
                 int root, MPI_Comm comm)
*/

MPI_Scatterv(d, counts, displs, MPI_INT, local, counts[rank], MPI_INT, MASTER, MPI_COMM_WORLD);

printArray(local, counts[rank]);

MPI_Finalize();

}

