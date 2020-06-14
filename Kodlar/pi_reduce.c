#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI kutuphanesi

#define MASTER 0
#define N 500000000

double f(double x) {
     return 4.0 / (1.0 + x * x);
}

double *create1DArray(int n) {
     double *T = (double *)malloc(n * sizeof(double));
     return T;
}

int main(void) {

int rank, size, i;

MPI_Init(NULL, NULL);

double t1, t2;

t1 = MPI_Wtime(); // start clock

MPI_Status status;

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int chunk = N / size;
double* mesh;
if (rank == MASTER) {
     mesh = create1DArray(N);   
     for (i = 1; i <= N; i++)
          mesh[i] = (double)i / (double)N;
}

double *local = create1DArray(chunk);

MPI_Scatter(mesh, chunk, MPI_DOUBLE, local, chunk, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

double sum = 0.0;

for (i = 0; i < chunk; i++)
     sum += f(local[i]);

printf("I am rank %d, my sum is %f\n", rank, sum);
// MPI_Barrier(MPI_COMM_WORLD);

double pi;
MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

printf("I am rank %d, my pi is %f\n", rank, pi/(double)N);

MPI_Barrier(MPI_COMM_WORLD);
t2 = MPI_Wtime(); // stop clock

printf("Elapsed time = %f sec.\n", t2 - t1);

MPI_Finalize();

return 0;

}
