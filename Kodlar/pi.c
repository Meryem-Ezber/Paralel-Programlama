#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "mpi.h" //MPI kutuphanesi

#define PI acos(-1)

#define N 500000000
#define MASTER 0

double f(double x) {
     return 4 / (1 + x * x); // return 1 / (1 + pow(x, 2));
}

int main(void) {

int rank, size;

MPI_Init(NULL, NULL);

double t1, t2;

t1 = MPI_Wtime(); // start clock

MPI_Status status;

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int i, chunkSize = N / size;

double toplam = 0.0;
double masterToplam = 0.0;

for (i = rank * chunkSize + 1; i <= (rank + 1) * chunkSize; i++)
     toplam += f((double)i / (double)N);

printf("rank = %d --> toplam = %f\n", rank, toplam);

if (rank != MASTER)
     MPI_Send(&toplam, 1, MPI_DOUBLE, MASTER, 777+rank, MPI_COMM_WORLD);
else {
     for (i = 1; i < size; i++) {
          MPI_Recv(&masterToplam, 1, MPI_DOUBLE, i, 777+i, MPI_COMM_WORLD, &status);
          toplam += masterToplam; 
     }

     printf("Appr. PI = %.16f\n", toplam / (double)N);
     printf("Exact PI = %.16f\n", PI);
}

MPI_Barrier(MPI_COMM_WORLD);
t2 = MPI_Wtime(); // stop clock

printf("Elapsed time = %f sec.\n", t2 - t1);

MPI_Finalize();

return 0;

}
