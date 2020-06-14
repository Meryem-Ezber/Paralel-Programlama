#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI kutuphanesi

#define MASTER 0
#define N 12

int *create1DArray(int n) {
     int *T = (int *)malloc(n * sizeof(int));
     return T;
}

void printArray(int *T, int n) {
     int i;
     for (i = 0; i < n; i++)
          printf("%d ", T[i]); 

     puts("");  
}  

int main(void) {

int rank, size, i;

MPI_Init(NULL, NULL);

MPI_Status status;

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int chunk = N / size;

int *array;  // each process has its own array address
int *local = create1DArray(chunk);

for (i = 0; i < chunk; i++)
    local[i] = 0;

if (rank == MASTER) {
     array = create1DArray(N); // only MASTER process stores array data  
     for (i = 0; i < N; i++)
          array[i] = i + 1;

     printArray(array, N);
}
MPI_Barrier(MPI_COMM_WORLD);


printf("rank %d printing...\n", rank);
printArray(local, chunk);

MPI_Barrier(MPI_COMM_WORLD);

MPI_Scatter(array, chunk, MPI_INT, local, chunk, MPI_INT, MASTER, MPI_COMM_WORLD);

printf("rank %d printing...\n", rank);
printArray(local, chunk);

MPI_Finalize();

return 0;

}
