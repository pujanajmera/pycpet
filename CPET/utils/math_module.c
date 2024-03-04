# include <stdio.h>
# include <stdlib.h>
# include<omp.h>
# include <math.h>
//# include <mpi.h>

// faster: gcc -fopenmp math_module.c -o math_module.so -shared -fPIC -O3 -march=native -funroll-loops -ffast-math
// compile without mpi w gcc -fopenmp math_module.c -o math_module.so -shared -fPIC
// compile w mpicc -fPIC -shared -o matmulmodule.so matmulmodule.c -fopenmp

void sparse_dot(double* ret, int* indptr, int indptrlen, int* indA, int lenindA, double* A, int lenA, double* B, int size_array){
    int rows = indptrlen - 1;
    int i; 
    int j;
    int provided = 0;
    int ptr;
    int ptrtemp;
    double sum;
    int numInRow;
    ptr = 0;
    
    // MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize 
    #pragma omp parallel shared(A, B, ret) private(i,j)
    {
        for(i=1; i < indptrlen; i++){
            sum = 0;
            numInRow= indptr[i] - indptr[i-1];
            ptrtemp = ptr;
            ptr += numInRow;


            for(j = ptrtemp; j < ptr; j++){
                    sum += A[j]*B[indA[j]];

            }
        
            ret[i-1] = sum; 

        }
    }
    // MPI_Finalize();
}


void dot(double* ret, double* A, double* B, int rows, int cols){
    int i; 
    int j;
    int provided = 0;
    double sum;
    //MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize 
    #pragma omp parallel shared(A, B, ret) private(i,j)
    {
        for(i=0; i < rows; i++){
            sum = 0;
            for(j = 0; j < cols; j++){
                    sum += A[i*cols + j]*B[j];

            }
            ret[i] = sum; 
        }
    }
    //MPI_Finalize();
}



void vecaddn(double* ret, double* A, double* B, int lenA){
    int i; 
    #pragma omp parallel shared(A, B, ret) private(i)
    {
    for (i = 0; i < lenA; i++) {
                ret[i] = A[i] + B[i];
            }
    }   
}
// create function for batched einsum_ij_i
// it intakes a 3D array and returns a 2D array





void einsum_operation_batch(int batch, int rows, double r_mag[batch][rows], double Q[rows], double R[batch][rows][3], double result[batch][3]){
    int i; 
    int j;
    int k;
    int provided = 0;
    double sum[3] = {0.0, 0.0, 0.0};
    double factor = 14.3996451;
    double r_0;
    double r_1;
    double r_2;
    double ele_1;
    double ele_2;
    double ele_3;
    double compute_singular;
    //MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize 
    #pragma omp parallel shared(R, r_mag, Q, result) private(i,j,k)
    {
        for(k=0; k < batch; k++){
            for(i=0; i < rows; i++){
                r_0 = R[k][i][0];
                r_1 = R[k][i][1];
                r_2 = R[k][i][2];
                ele_1 = r_mag[k][i];
                ele_2 = Q[i];
                ele_3 = r_mag[k][i];
                compute_singular = factor * ele_1 * ele_2 * ele_3;
                sum[0] += compute_singular * r_0;
                sum[1] += compute_singular * r_1;
                sum[2] += compute_singular * r_2;

            }
            result[k][0] = sum[0];
            result[k][1] = sum[1];
            result[k][2] = sum[2];
        }
    }
    //MPI_Finalize();
}

void einsum_operation(int rows, double r_mag[rows], double Q[rows], double R[rows][3], double result[3]){
    int i; 
    int j;
    int provided = 0;
    double sum[3] = {0.0, 0.0, 0.0};
    double factor = 14.3996451;
    double r_0;
    double r_1;
    double r_2;
    double ele_1;
    double ele_2;
    double ele_3;
    double compute_singular;
    //MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize 
    #pragma omp parallel shared(R, r_mag, Q, result) private(i,j)
    {
        for(i=0; i < rows; i++){
            r_0 = R[i][0];
            r_1 = R[i][1];
            r_2 = R[i][2];
            ele_1 = r_mag[i];
            ele_2 = Q[i];
            ele_3 = r_mag[i];
            compute_singular = factor * ele_1 * ele_2 * ele_3;
            sum[0] += compute_singular * r_0;
            sum[1] += compute_singular * r_1;
            sum[2] += compute_singular * r_2;

        }
        result[0] = sum[0];
        result[1] = sum[1];
        result[2] = sum[2];
    }
    //MPI_Finalize();
}



// ret is a 3D array
// A is a 3D array
// batch is the number of 2D arrays in the 3D array
// rows is the number of rows in each 2D array
// cols is the number of columns in each 2D array
void einsum_ij_i_batch(int batch, int rows, int cols, double A[batch][rows][cols], double ret[batch][rows]){
    int i; 
    int j;
    int k;
    int provided = 0;
    double sum;
    //MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize 
    #pragma omp parallel shared(A, ret) private(i,j,k)
    {
        for(k=0; k < batch; k++){
            for(i=0; i < rows; i++){
                sum = 0;
                for(j = 0; j < cols; j+=1){
                        sum += A[k][i][j];
                }
                ret[k][i] = sum;    
            }
        }
    }
    //MPI_Finalize();
}


void einsum_ij_i(int rows, int cols, double A[rows][cols], double ret[rows]){
    int i; 
    int j;
    int provided = 0;
    double sum;
    //MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize
    #pragma omp parallel shared(A, ret) private(i,j)
    {
    
        for(i=0; i < rows; i++){
            sum = 0;
            for(j = 0; j < cols; j+=1){
                    sum += A[i][j];
            }
            ret[i] = sum;    
        }
    }
}
