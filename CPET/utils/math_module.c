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


void cross_product(double *u, double *v, double *product)
{
        double p1 = u[1]*v[2]
                - u[2]*v[1];

        double p2 = u[2]*v[0]
                - u[0]*v[2];

        double p3 = u[0]*v[1]
                - u[1]*v[0];

        product[0] = p1;
        product[1] = p2;
        product[2] = p3;

}


void norm(double *u, double *norm_val)
{
    *norm_val = sqrt(
        pow(u[0], 2) + 
        pow(u[1], 2) + 
        pow(u[2], 2)
    );
}


double euclidean_dist(double* x_0, double* x_1){
    double sum = 0;
    for(int i = 0; i < 3; i++){
        sum += pow(x_0[i] - x_1[i], 2);
    }
    return sqrt(sum);
}


double curve(double* x_0, double* x_1){
    // compute cross of x_0 and x_1
    double cross_prod[3];
    double norm_cross_prod;
    double norm_denom;

    cross_product(x_0, x_1, cross_prod);
    norm(cross_prod, &norm_cross_prod);
    norm(x_0, &norm_denom);
    double curve_ret = norm_cross_prod / pow(norm_denom, 3);

    return curve_ret;

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


void calc_field(double E[3], double x_init[3], int n_charges, double x[n_charges][3], double Q[n_charges]){
    // calculate the field

    // subtract x_init from x
    double R[n_charges][3];
    double r_mag[n_charges];
    double E_array[3];
    double r_sq[n_charges][3];
    double r_mag_sq[n_charges];
    
    // make R a 2D array
    for (int j = 0; j < 3; j++)
    {
        double x_init_single = x_init[j];
        # pragma omp parallel for
        for (int i = 0; i < n_charges; i++)
        {
            R[i][j] = x[i][j] - x_init_single;
            r_sq[i][j] = pow(R[i][j], 2);
        }
    }
    
    einsum_ij_i(n_charges, 3, r_sq, r_mag_sq); // this might be a funky shape
    
    # pragma omp parallel for
    for (int i = 0; i < n_charges; i++)
    {
        // elementwise raise to -3/2
        r_mag[i] = pow(r_mag_sq[i], -1.5); // if this is zero then we have a problem
    }
    
    // compute einsum operation
    einsum_operation(n_charges, r_mag, Q, R, E_array);

    // trivial step
    double factor = 14.3996451;
    for (int i = 0; i < 3; i++)
    {
        E[i] = E_array[i] * factor;
    }


}


void propagate_topo(double result[3], double x_init[3], int n_charges, double x[n_charges][3], double Q[n_charges], double step_size){
    // propagate the topology
    double epsilon = 10e-7;
    double E[3];
    double E_norm;

    calc_field(E, x_init, n_charges, x, Q);
    norm(E, &E_norm);
    for (int i = 0; i < 3; i++)
    {
        result[i] = x_init[i] + step_size * E[i] / (E_norm + epsilon);
    }
}


void thread_operation(int n_charges, int n_iter, double step_size, double x_0[3], double dimensions[3], double x[n_charges][3], double Q[n_charges], double ret[2]){
    bool bool_inside = true;
    int count = 0; 
    double x_init[3] = {x_0[0], x_0[1], x_0[2]};
    double x_overwrite[3] = {x_0[0], x_0[1], x_0[2]};
    
    for (int i = 0; i < n_iter; i++) {

        propagate_topo(x_overwrite, x_overwrite, n_charges, x, Q, step_size);
        // overwrite x_init with x_overwrite

        double half_length = dimensions[0];
        double half_width = dimensions[1];
        double half_height = dimensions[2];
        

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
            count += 1;
            //printf("Breaking out of loop\n");
            break;
        }
    }

    double x_init_plus[3];
    double x_init_plus_plus[3];
    double x_0_plus[3];
    double x_0_plus_plus[3];    

    propagate_topo(x_init_plus, x_init, n_charges, x, Q, step_size);
    propagate_topo(x_init_plus_plus, x_init_plus, n_charges, x, Q, step_size);
    propagate_topo(x_0_plus, x_overwrite, n_charges, x, Q, step_size);
    propagate_topo(x_0_plus_plus, x_0_plus, n_charges, x, Q, step_size);

    double curve_arg_1[3];
    double curve_arg_2[3];
    double curve_arg_3[3];
    double curve_arg_4[3];
    
    for (int i = 0; i < 3; i++) {
        curve_arg_1[i] = x_init_plus[i] - x_init[i];
        curve_arg_2[i] = x_init_plus_plus[i] - 2* x_init_plus[i] + x_init[i];
        curve_arg_3[i] = x_0_plus[i] - x_overwrite[i];
        curve_arg_4[i] = x_0_plus_plus[i] - 2* x_0_plus[i] + x_overwrite[i];
    }

    double curve_init = curve(curve_arg_1, curve_arg_2);
    double curve_final = curve(curve_arg_3, curve_arg_4);
    double curve_mean = (curve_init + curve_final) / 2;
    double dist = euclidean_dist(x_init, x_overwrite);


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
