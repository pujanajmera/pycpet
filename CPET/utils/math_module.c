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


void einsum_ij_i(double *ret, double *A, int rows, int cols){
    int i; 
    int j;
    int provided = 0;
    double sum;
    int threads;
    //MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize 
    #pragma omp parallel shared(A, ret) private(i,j)
        #pragma omp for reduction(+:sum) schedule(static)
            for(i=0; i < rows; i++){
                sum = 0;
                for(j = 0; j < cols; j+=1){
                        sum += A[i*cols + j];
                }
                ret[i] = sum;    
            }
    //MPI_Finalize();
}



void einsum_operation(double* R, double *r_mag, double* Q, int n, double* result) {
    double sum[3] = {0.0, 0.0, 0.0};
    double factor = 14.3996451;
    double r_0;
    double r_1;
    double r_2;
    double ele_1;
    double ele_2;
    double ele_3;
    double compute_singular;
    //printf("einsum_operation: %f\n", r_mag[0]);
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (int i = 0; i < n; ++i) {        
        //pow_val = (r_mag[i] != 0) ? pow(r_mag[i], 3) : small_value;
        
        //compute_singular = pow(r_mag[i], 1) 
        compute_singular = r_mag[i] * Q[i];
        
        r_0 = R[i*3];
        r_1 = R[i*3 + 1];
        r_2 = R[i*3 + 2];

        ele_1 = r_0 * compute_singular;
        ele_2 = r_1 * compute_singular;
        ele_3 = r_2 * compute_singular;

        //if (i < 3){
            // print Rvals
            //printf("Rvals: %f, %f, %f\n", r_0, r_1, r_2);
            //printf("Qvals: %f\n", compute_singular);
            //printf("ele_1: %f, ele_2: %f, ele_3: %f\n", ele_1 * factor, ele_2 * factor, ele_3 * factor);
        //}

        sum[0] += ele_1 * factor;
        sum[1] += ele_2 * factor;
        sum[2] += ele_3 * factor;
        
    }
    //printf("sum: %f, %f, %f\n", sum[0], sum[1], sum[2]);
    result[0] = sum[0];
    result[1] = sum[1];
    result[2] = sum[2];
    //printf("result: %f, %f, %f\n", result[0], result[1], result[2]);
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

