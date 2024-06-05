#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void generate_random_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = (double)rand() / RAND_MAX;
        }
    }
}

void print_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.2f\t", matrix[i * size + j]);
        }
        printf("\n");
    }
}

void multiply_matrices(double *A, double *B, double *C, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            C[i * dim + j] = 0;
            for (int k = 0; k < dim; k++) {
                C[i * dim + j] += A[i * dim + k] * B[k * dim + j];
            }
        }
    }
}

int main(int argc, char *argv[]){
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int matrix_dim = sqrt(size);
    if (matrix_dim * matrix_dim != size) {
        if (rank == 0) {
            printf("Number of processes must be a perfect square\n");
        }
        MPI_Finalize();
        return 1;
    }

    double *matrix_A = NULL;
    double *matrix_B = NULL;
    double *matrix_C = NULL;
    double *matrix_C_truth = NULL;
    double *diagonal = NULL;

    if (rank == 0) {
        srand(time(NULL));
        matrix_A = (double *)calloc(matrix_dim * matrix_dim, sizeof(double));
        matrix_B = (double *)calloc(matrix_dim * matrix_dim, sizeof(double));
        matrix_C = (double *)calloc(matrix_dim * matrix_dim, sizeof(double));
        matrix_C_truth = (double *)calloc(matrix_dim * matrix_dim, sizeof(double));
        generate_random_matrix(matrix_A, matrix_dim);
        generate_random_matrix(matrix_B, matrix_dim);
        printf("Matrix A:\n");
        print_matrix(matrix_A, matrix_dim);
        printf("Matrix B:\n");
        print_matrix(matrix_B, matrix_dim);
        printf("Matrix Ground Truth Calculation:\n");
        multiply_matrices(matrix_A, matrix_B, matrix_C_truth, matrix_dim);
        print_matrix(matrix_C_truth, matrix_dim);
    }

    double local_b_cell;
    MPI_Scatter(matrix_B, 1, MPI_DOUBLE, &local_b_cell, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for(int iter = 0; iter < matrix_dim; iter++)
    {

        if (rank == 0){
            diagonal = (double *)calloc(matrix_dim, sizeof(double));
            for (int i = 0; i < matrix_dim; i++) {
                int loc = (i + iter) < matrix_dim ? i * matrix_dim + i + iter : i * matrix_dim + (i + iter - matrix_dim);
                diagonal[i] = matrix_A[loc];
            }
        }

        // Create array to hold diagonal elements to be received by each rank
        double local_a_cell;

        // Prepare the displacements and send counts for MPI_Scatterv
        int *sendcounts = NULL;
        int *displs = NULL;
        if (rank == 0) {
            sendcounts = (int *)calloc(size, sizeof(int));
            displs = (int *)calloc(size, sizeof(int));
            for (int i = 0; i < size; i++) {
                sendcounts[i] = 1;
                displs[i] = i / matrix_dim;
            }
        }

        MPI_Scatterv(diagonal, sendcounts, displs, MPI_DOUBLE, &local_a_cell, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double local_c_cell;

        local_c_cell = local_a_cell * local_b_cell;

        if (rank != 0) {
                MPI_Send(&local_c_cell, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        } else {
            matrix_C[0] += local_c_cell;
            for (int i = 1; i < size; i++) {
                int row = i / matrix_dim;
                int col = i % matrix_dim;
                MPI_Recv(&local_c_cell, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                matrix_C[row * matrix_dim + col] += local_c_cell;
            }
        }
        double send_data = local_b_cell;
        double recv_data;

        int to = (rank - matrix_dim >= 0) ? rank - matrix_dim : size - matrix_dim + rank;
        int from = (rank + matrix_dim < size) ? rank + matrix_dim : rank - size + matrix_dim;

        MPI_Send(&send_data, 1, MPI_DOUBLE, to, 0, MPI_COMM_WORLD);

        MPI_Recv(&recv_data, 1, MPI_DOUBLE, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        local_b_cell = recv_data;


    }

    
    if (rank == 0) {
        free(matrix_A);
        free(matrix_B);
        printf("Matrix C Parallel Calculation:\n");
        print_matrix(matrix_C, matrix_dim);
        free(matrix_C);
        free(matrix_C_truth);
    }

    MPI_Finalize();
    return 0;
}
