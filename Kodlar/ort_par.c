#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI kutuphanesi

#define N 5000000
#define MASTER 0

double *create1DArray(int n) {
     double *T = (double *)malloc(n * sizeof(double));
     return T;
}

int main(void) {

int rank, size, i;

MPI_Init(NULL, NULL);

MPI_Status status;

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int chunk = N / size;

double local_toplam = 0.0;
double toplam = 0.0;

// Initialization
double *dizi;
double *local_dizi;

if (rank == MASTER) {
    dizi = create1DArray(N);
    local_dizi = create1DArray(chunk);

    for (i = 0; i < N; i++)
        dizi[i] = 1.0;

} else {
    local_dizi = create1DArray(chunk);
}

// Scatter (data distribution)
MPI_Scatter(dizi, chunk, MPI_DOUBLE, local_dizi, chunk, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

// Local computation
for (i = 0; i < chunk; i++)
    local_toplam += local_dizi[i];
    
// Gather data with reduction
MPI_Reduce(&local_toplam, &toplam, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

// MASTER prints out the average
if (rank == MASTER)
    printf("Ortalama = %f\n", toplam / N);

MPI_Finalize();

}







