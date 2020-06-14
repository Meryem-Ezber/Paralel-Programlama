#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI kutuphanesi

#define ROW 8
#define COL 5

#define MASTER 0

int *create1DArray(int n) {
     int *T = (int *)malloc(n * sizeof(int));
     return T;
}

int **create2DArray(int n1, int n2) {
     int i;
     int **T = (int **)malloc(n1 * sizeof(int *));
     for (i = 0; i < n1; i++)
          T[i] = (int *)malloc(n2 * sizeof(int));
          // T[i] = create1DArray(n2);
     return T;
}

void fillMatrix(int **T) {
     int i, j;
     for (i = 0; i < ROW; i++)
          for (j = 0; j < COL; j++)
               T[i][j] = i * COL + j + 1;
}

void printMatrix(int **T) {
     int i, j;
     for (i = 0; i < ROW; i++) {
          for (j = 0; j < COL; j++) {
               printf("%d ", T[i][j]);
          }
          printf("\n");
     }
}

void printArray(int *T) {
     int i;
     for (i = 0; i < ROW; i++)
          printf("%d ", T[i]); 

     puts("");  
}  

int main(void) {

int rank, size, i;

MPI_Init(NULL, NULL);

MPI_Status status;

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// int **matris = create2DArray(ROW, COL);
int *array = create1DArray(ROW);

if (rank == MASTER) {
     for (i = 0; i < ROW; i++)
          array[i] = i + 1;
}

printf("rank %d printing...\n", rank);
printArray(array);
MPI_Barrier(MPI_COMM_WORLD);

MPI_Bcast(array, ROW, MPI_INT, MASTER, MPI_COMM_WORLD);

printf("rank %d printing...\n", rank);
printArray(array);

MPI_Finalize();

return 0;

}
