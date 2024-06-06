# include <stdio.h>
# include <stdlib.h>
# include <stdbool.h>
# include<omp.h>
# include <math.h>
//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_blas.h>
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
    //#pragma omp parallel shared(A, B, ret) private(i,j)
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


void cross_product(float *u, float *v, float *product)
{
        float p1 = u[1]*v[2]
                - u[2]*v[1];

        float p2 = u[2]*v[0]
                - u[0]*v[2];

        float p3 = u[0]*v[1]
                - u[1]*v[0];

        product[0] = p1;
        product[1] = p2;
        product[2] = p3;

}


void norm(float *u, float *norm_val)
{
    *norm_val = sqrt(
        pow(u[0], 2) + 
        pow(u[1], 2) + 
        pow(u[2], 2)
    );
}


double euclidean_dist(float* x_0, float* x_1){
    float sum = 0;
    for(int i = 0; i < 3; i++){
        sum += pow(fabsf(x_0[i] - x_1[i]), 2);
    }
    return sqrt(sum);
}


double curve(float* x_0, float* x_1){
    // compute cross of x_0 and x_1
    float cross_prod[3];
    float norm_cross_prod;
    float norm_denom;

    cross_product(x_0, x_1, cross_prod);
    norm(cross_prod, &norm_cross_prod);
    norm(x_0, &norm_denom);
    float curve_ret = norm_cross_prod / pow(norm_denom, 3);

    return curve_ret;

}


void vecaddn(float* ret, float* A, float* B, int lenA){
    int i; 
    //#pragma omp parallel shared(A, B, ret) private(i)
    {
    for (i = 0; i < lenA; i++) {
                ret[i] = A[i] + B[i];
            }
    }   
}


void einsum_operation_batch(int batch, int rows, float r_mag[batch][rows], float Q[rows], float R[batch][rows][3], float result[batch][3]){
    int i; 
    int j;
    int k;
    int provided = 0;
    float sum[3] = {0.0, 0.0, 0.0};
    float factor = 14.3996451;
    float r_0;
    float r_1;
    float r_2;
    float ele_1;
    float ele_2;
    float ele_3;
    float compute_singular;
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


void einsum_operation(int rows, float r_mag[rows], float Q[rows], float R[rows][3], float result[3]){
    int i; 
    int j;
    int provided = 0;
    float sum[3] = {0.0, 0.0, 0.0};
    float factor = 14.3996451;
    float r_0;
    float r_1;
    float r_2;
    float ele_1;
    float ele_2;
    float ele_3;
    float compute_singular;
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


void einsum_ij_i_batch(int batch, int rows, int cols, float A[batch][rows][cols], float ret[batch][rows]){
    int i; 
    int j;
    int k;
    int provided = 0;
    float sum;
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


void einsum_ij_i(int rows, int cols, float A[rows][cols], float ret[rows]){
    int i; 
    int j;
    int provided = 0;
    float sum;
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


void calc_field(float E[3], float x_init[3], int n_charges, float x[n_charges][3], float Q[n_charges]){
    // calculate the field

    // subtract x_init from x
    float R[n_charges][3];
    float r_mag[n_charges];
    float E_array[3];
    float r_sq[n_charges][3];
    float r_mag_sq[n_charges];
    
    // make R a 2D array
    for (int j = 0; j < 3; j++)
    {
        float x_init_single = x_init[j];
        //# pragma omp parallel for
        for (int i = 0; i < n_charges; i++)
        {
            R[i][j] = x[i][j] - x_init_single;
            r_sq[i][j] = pow(R[i][j], 2);
        }
    }
    
    einsum_ij_i(n_charges, 3, r_sq, r_mag_sq); // this might be a funky shape
    
    //# pragma omp parallel for
    for (int i = 0; i < n_charges; i++)
    {
        // elementwise raise to -3/2
        r_mag[i] = pow(r_mag_sq[i], -1.5); // if this is zero then we have a problem
    }
    
    // compute einsum operation
    einsum_operation(n_charges, r_mag, Q, R, E_array);

    // trivial step
    float factor = 14.3996451;
    for (int i = 0; i < 3; i++)
    {
        E[i] = E_array[i] * factor;
    }


}

void calc_field_base(float E[3], float x_init[3], int n_charges, float x[n_charges][3], float Q[n_charges]){
    // calculate the field

    // subtract x_init from x
    float R[n_charges][3];
    float r_mag[n_charges];
    float r_sq[n_charges][3];
    float r_mag_sq[n_charges];
    float factor = 14.3996451;
    float r_norm[n_charges];
    
    //# pragma omp parallel for
    
    # pragma omp parallel for
    for (int i = 0; i < n_charges; i++)
    {
        R[i][0] = x[i][0] - x_init[0];
        R[i][1] = x[i][1] - x_init[1];
        R[i][2] = x[i][2] - x_init[2];
        r_norm[i] = sqrt(pow(R[i][0], 2) + pow(R[i][1], 2) + pow(R[i][2], 2));
        r_mag[i] = pow(r_norm[i], -3);

        E[0] += factor * r_mag[i] * Q[i] * R[i][0];
        E[1] += factor * r_mag[i] * Q[i] * R[i][1];
        E[2] += factor * r_mag[i] * Q[i] * R[i][2];

    }



}


void propagate_topo(float result[3], float x_init[3], int n_charges, float x[n_charges][3], float Q[n_charges], float step_size){
    // propagate the topology
    float E[3] = {0.0, 0.0, 0.0};
    float E_norm;
    //printf("propagating!!");

    //calc_field(E, x_init, n_charges, x, Q);
    calc_field_base(E, x_init, n_charges, x, Q);
    
    norm(E, &E_norm);
    for (int i = 0; i < 3; i++)
    {
        result[i] = x_init[i] + step_size * E[i] / (E_norm);
    }
}


void thread_operation(int n_charges, int n_iter, float step_size, float x_0[3], float dimensions[3], float x[n_charges][3], float Q[n_charges], float ret[2]){
    bool bool_inside = true;
    float x_init[3] = {x_0[0], x_0[1], x_0[2]};
    float x_overwrite[3] = {x_0[0], x_0[1], x_0[2]};
    // print x_init
    // printf("x_init %f %f %f\n", x_init[0], x_init[1], x_init[2]);
    int i; 
    float half_length = dimensions[0];
    float half_width = dimensions[1];
    float half_height = dimensions[2];
    
    for (i = 0; i < n_iter; i++) {

        propagate_topo(x_overwrite, x_overwrite, n_charges, x, Q, step_size);
        // overwrite x_init with x_overwrite

        if (
            x_overwrite[0] < -half_length || 
            x_overwrite[0] > half_length || 
            x_overwrite[1] < -half_width || 
            x_overwrite[1] > half_width || 
            x_overwrite[2] < -half_height || 
            x_overwrite[2] > half_height){
            bool_inside = false;
        }

        // printf("%f %f %f\n", x_overwrite[0], x_overwrite[1], x_overwrite[2]);

        if (!bool_inside){
            // printf("Breaking out of loop at iteration: %i\n", i);
            break;
        }
        // printf("x_final @ iter %i out of %i %f %f %f\n", i, n_iter, x_overwrite[0], x_overwrite[1], x_overwrite[2]);
        
    }
        
    
    float x_init_plus[3];
    float x_init_plus_plus[3];
    float x_0_plus[3];
    float x_0_plus_plus[3];    

    propagate_topo(x_init_plus, x_init, n_charges, x, Q, step_size);
    propagate_topo(x_init_plus_plus, x_init_plus, n_charges, x, Q, step_size);
    propagate_topo(x_0_plus, x_overwrite, n_charges, x, Q, step_size);
    propagate_topo(x_0_plus_plus, x_0_plus, n_charges, x, Q, step_size);

    float curve_arg_1[3];
    float curve_arg_2[3];
    float curve_arg_3[3];
    float curve_arg_4[3];
    
    for (int i = 0; i < 3; i++) {
        curve_arg_1[i] = x_init_plus[i] - x_init[i];
        curve_arg_2[i] = x_init_plus_plus[i] - 2* x_init_plus[i] + x_init[i];
        curve_arg_3[i] = x_0_plus[i] - x_overwrite[i];
        curve_arg_4[i] = x_0_plus_plus[i] - 2* x_0_plus[i] + x_overwrite[i];
    }

    float curve_init = curve(curve_arg_1, curve_arg_2);
    float curve_final = curve(curve_arg_3, curve_arg_4);
    float curve_mean = (curve_init + curve_final) / 2;
    float dist = euclidean_dist(x_0, x_overwrite);
    

    ret[0] = dist;
    ret[1] = curve_mean;

}

/*
void vecaddn_gsl(gsl_vector* ret, gsl_vector* A, gsl_vector* B, int lenA){
    int i; 
    #pragma omp parallel shared(A, B, ret) private(i)
    {
    for (i = 0; i < lenA; i++) {
                gsl_vector_set(ret, i, gsl_vector_get(A, i) + gsl_vector_get(B, i));
            }
    }   
}




void cross_product_gsl(const gsl_vector *u, const gsl_vector *v, gsl_vector *product)
{
        double p1 = gsl_vector_get(u, 1)*gsl_vector_get(v, 2)
                - gsl_vector_get(u, 2)*gsl_vector_get(v, 1);

        double p2 = gsl_vector_get(u, 2)*gsl_vector_get(v, 0)
                - gsl_vector_get(u, 0)*gsl_vector_get(v, 2);

        double p3 = gsl_vector_get(u, 0)*gsl_vector_get(v, 1)
                - gsl_vector_get(u, 1)*gsl_vector_get(v, 0);

        gsl_vector_set(product, 0, p1);
        gsl_vector_set(product, 1, p2);
        gsl_vector_set(product, 2, p3);
}

void norm_gsl(const gsl_vector *u, double *norm_val)
{
    *norm_val = sqrt(
        pow(gsl_vector_get(u, 0), 2) + 
        pow(gsl_vector_get(u, 1), 2) + 
        pow(gsl_vector_get(u, 2), 2)
    );
}

double euclidean_dist_gsl(gsl_vector* x_0, gsl_vector* x_1){
    double sum = 0;
    for(int i = 0; i < 3; i++){
        sum += pow(gsl_vector_get(x_0, i) - gsl_vector_get(x_1, i), 2);
    }
    return sqrt(sum);
}

double curve_gsl(gsl_vector* x_0, gsl_vector* x_1){
    // compute cross of x_0 and x_1
    gsl_vector* cross_prod = gsl_vector_alloc(3);
    double norm_cross_prod;
    double norm_denom;

    cross_product(x_0, x_1, cross_prod);
    norm(cross_prod, &norm_cross_prod);
    norm(x_0, &norm_denom);
    double curve_ret = norm_cross_prod / pow(norm_denom, 3);

    // free memory
    gsl_vector_free(cross_prod);
    return curve_ret;

}

*/
