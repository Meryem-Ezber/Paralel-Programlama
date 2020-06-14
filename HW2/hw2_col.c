#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define MASTER 0

float *create1DArray(int n) {
    float *T = (float *)malloc(n * sizeof(float));
    return T;
}

void fillArray(float *T, int n) {
    int i;
    for (i = 0; i < n; i++)
        T[i] = 1.0;
}

void printArray(float *T, int n) {
    int i;
    for (i = 0; i < n; i++)
        printf("%.2f ", T[i]);
    puts("");
}

float innerProd(float *u, float *v, int n) {
    int i;
    float sum = 0.0;
    for (i = 0; i < n; i++)
        sum += u[i] * v[i];
    return sum;
}

float *mat_vec_mult(float *M, float *v, int n1, int n2) {
    int i;
    float *r = create1DArray(n1);
    for (i = 0; i < n1; i++)
        r[i] = innerProd(&M[i*n2], v, n2);
    return r;
}

int main(int argc, char *argv[]) {

int n1 = 30000;
int n2 = 30000;

int rank, size, i;
double t_start, t_finish;
double t_start_comm, t_finish_comm;
double t_start_comp, t_finish_comp;

MPI_Init(NULL, NULL);

// start timer
t_start = MPI_Wtime();

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int chunk = n1 / size;

float *A, *x, *b, *A_local, *b_local;

x = create1DArray(n2);
A_local = create1DArray(chunk * n2);
b_local = create1DArray(chunk);

if (rank == MASTER) {
    A = create1DArray(n1 * n2);
    fillArray(A, n1 * n2);
    fillArray(x, n2);
    //printArray(A,n1*n2);
    //printArray(x,n2);
    b = create1DArray(n1);
}
    
    
// Yeni data tipi tanımlandı : colType
MPI_Datatype colType, newColType;

int blocklength = 1, stride = n2, count = n1;
MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &colType);
MPI_Type_commit(&colType);

MPI_Type_create_resized(colType, 0, 1*sizeof(float), &newColType);
MPI_Type_commit(&newColType);

// T comm: master ile iletişim için harcanan zaman miktarı
t_start_comm = MPI_Wtime();

// broadcast x vector to all processes
MPI_Bcast(x, n2, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

// scatter A to each process
MPI_Scatter(A, chunk, newColType, A_local, chunk*n1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

// local computation
// T comp: bilgi işlem görevleri için harcanan zaman miktarı
t_start_comp = MPI_Wtime();
b_local = mat_vec_mult(A_local, x, chunk, n2);
t_finish_comp = MPI_Wtime();
    
// gather b_local's into MASTER's b
MPI_Gather(b_local, chunk, MPI_FLOAT, b, chunk, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    
t_finish_comm = MPI_Wtime();

// end timer
t_finish = MPI_Wtime();

if (rank == MASTER) {
    //printArray(b, n1);
    printf("Time elapsed = %f sec.\n", t_finish - t_start);
    printf("Communication time = %f sec.\n", t_finish_comm - t_start_comm);
    printf("Computing time = %f sec.\n", t_finish_comp - t_start_comp);
}

MPI_Finalize();

}












