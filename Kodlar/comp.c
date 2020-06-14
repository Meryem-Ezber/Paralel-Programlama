#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI kutuphanesi
#include <math.h>
#include <time.h>

#define N 3000000
#define MASTER 0

double *create1DArray(int n) {
     double *T = (double *)malloc(n * sizeof(double));
     return T;
}

int main(void) {
    
    srand(time(NULL));
    
    double t1, t2;

    int rank, size, i;

    MPI_Init(NULL, NULL);
    
    t1 = MPI_Wtime();

    MPI_Status status;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int chunk = N / size;

    double local_toplam = 0.0;
    double toplam = 0.0;
    double var_toplam = 0.0;
    double var_local = 0.0;

    // Initialization
    double *dizi;
    double *local_dizi;

    if (rank == MASTER) {
        dizi = create1DArray(N);
        local_dizi = create1DArray(chunk);
        for (i = 0; i < N; i++){
            dizi[i] = rand()%25;
        }
    } else {
        local_dizi = create1DArray(chunk);
    }

    // Scatter (data distribution)
    MPI_Scatter(dizi, chunk, MPI_DOUBLE, local_dizi, chunk, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    // Local computation
    for (i = 0; i < chunk; i++)
        local_toplam += local_dizi[i];
    
    printf("I am rank %d, my local toplam is %f\n", rank, local_toplam);
    
    // Gather data with reduction
    MPI_Reduce(&local_toplam, &toplam, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    
    //printf("toplam %f\n", toplam);

    float ortalama = toplam / N;

    // MASTER prints out the average
    if (rank == MASTER)
        printf("Ortalama = %f\n", ortalama);
    
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&ortalama, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    
    // herkese aynı ortalama gönderildi mi kontrolü
    // printf("I am rank %d, my average is %f\n", rank, ortalama);
        
    // Local var hesaplaması
    double temp = 0;
    // Local computation
    for (i = 0; i < chunk; i++){
        temp = local_dizi[i]-ortalama;
        var_local += pow(temp,2);
    }
       
    var_local = var_local / (N-1);
    
    // her rank kendi var'ını yazıyor
    printf("I am rank %d, my var is %f\n", rank, var_local);
        
    // her rank localindeki var'ı master rank a gönderiyor, master rank bu değerleri topluyor
    MPI_Reduce(&var_local, &var_toplam, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    
    // master rank standart sapmayı hesaplayıp ekrana basıyor
    if (rank == MASTER){
        printf("Standard deviation : %f\n", sqrt(var_toplam));
    }
        
    t2 = MPI_Wtime();
    
    printf("MPI_Wtime : %1.4f\n", t2-t1);
    fflush(stdout);
    
    MPI_Finalize();

}


