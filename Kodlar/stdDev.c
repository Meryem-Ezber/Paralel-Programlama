#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI kutuphanesi

#define MASTER 0
#define N 1000000

float *create1DArray(int n) {
     float *T = (float *)malloc(n * sizeof(float));
     return T;
}

void printArray(float *T, int n) {
     int i;
     for (i = 0; i < n; i++)
          printf("%f ", T[i]); 

     puts("");  
}

float topla(float *dd, int n) {
    int i;
    float top = 0.0;
    for (i = 0; i < n; i++)
        top += dd[i];

    return top;
}


int main(void) {

int rank, size, i;

MPI_Init(NULL, NULL);

MPI_Status status;

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int chunk = N / size;

float* x;
float* local_x = create1DArray(chunk);


if (rank == MASTER) {
     x = create1DArray(N);
     for (i = 0; i < N; i++)
          x[i] = 1.0;
}

// 1st step
MPI_Scatter(x, chunk, MPI_FLOAT, local_x, chunk, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

// 2nd step
float local_top = topla(local_x, chunk);

// 3rd step
float ort;
MPI_Reduce(&local_top, &ort, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
ort /= (float)N;

printf("Ortalama = %f\n", ort);

MPI_Finalize();

return 0;

}
